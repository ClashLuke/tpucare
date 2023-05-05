import datetime
import inspect
import json
import logging
import multiprocessing
import os
import signal
import subprocess
import tempfile
import threading
import time
import types
import typing
from contextlib import nullcontext
from typing import Callable


def call(cmd: str) -> str:
    return subprocess.check_output(cmd.split()).rstrip().decode()


class CallReturnError(ValueError):
    pass


def retry_call(cmd: typing.List[str], retries: int = -2):
    while retries != -1:
        retries -= 1
        popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
        out = ""
        for stdout_line in iter(popen.stdout.readline, ""):
            out += f'{out}'
            log(stdout_line.rstrip(), log_level=logging.DEBUG)
        popen.stdout.close()
        if not popen.wait():
            return out
    raise ValueError


PROJECT = call("gcloud config get project")
TPU_CMD = "gcloud alpha compute tpus tpu-vm"
GLOBAL_DICT = {}
CACHE_TIME = 10
LOG_LEVEL = logging.INFO
Context = typing.TypeVar("Context")
All = typing.Literal["all"]
SliceIndex = typing.Union[All, int]


def log(*message, log_level=1e9):
    if log_level > LOG_LEVEL:
        print(f'{datetime.datetime.now()} | {" ".join(map(str, message))}', flush=True)


def exec_command(repository: str, wandb_key: typing.Optional[str] = None, branch: str = "main",
                 setup_command: str = "(bash setup.sh; exit 0)", run_command: str = "bash run.sh",
                 install_python: bool = True):
    path = repository.split('/')[-1]
    if path.endswith('.git'):
        path = path[:-len('.git')]
    script = []
    if install_python:
        script.append("sudo apt-get -o DPkg::Lock::Timeout=-1 update")
        script.append("sudo apt-get -o DPkg::Lock::Timeout=-1 --fix-missing --fix-broken install -y git python3 "
                      "python3-pip")
    script.append(f"(rm -rf {path} ; pkill -f python3 ; exit 0)")
    script.append(f"python3 -m pip install --upgrade pip")
    script.append(f"git clone --depth 1 --branch {branch} {repository}")
    script.append(f"cd {path}")
    if wandb_key is not None:
        script.append("python3 -m pip install --upgrade wandb typer click")
        script.append(f"/home/ubuntu/.local/bin/wandb login {wandb_key}")
    script.extend([setup_command, f'screen -dmS model bash -c "cd {path} ; {run_command}"'])
    return ' &&\\\n'.join(script)


def retry_delete(host: str, zone: str, cmd: typing.List[str], retries: int = -2):
    try:
        return retry_call(cmd, retries)
    except CallReturnError:
        delete_one_tpu(host, host, zone)
        return ""


def log_entry(prefix: str, fmt: Callable):
    def _fn(fn: Callable):
        signature = inspect.signature(fn).parameters

        def _inner(*args, **kwargs):
            for a, s in zip(args, signature.keys()):
                kwargs[s] = a
            txt = f"{prefix} {fmt(kwargs)}"
            log(f"{txt} ...", log_level=logging.INFO)
            start_time = time.time()
            out = fn(**kwargs)
            log(f"finished {txt} after {time.time() - start_time:.1f}s", log_level=logging.INFO)
            return out

        return _inner

    return _fn


@log_entry("sending", lambda x: f"'{x['filename_on_tpu']}'")
def send_to_tpu(host: str, zone: str, filename_on_tpu: str, command: str, worker: SliceIndex = 0):
    with tempfile.NamedTemporaryFile(mode='w+') as f:
        f.write(command)
        f.flush()
        cmd = TPU_CMD.split(' ') + ["scp", f.name, f"ubuntu@{host}:~/{filename_on_tpu}", "--zone", zone, "--worker",
                                    str(worker)]
        retry_delete(host, zone, cmd, 4)


@log_entry("running", lambda x: f"'{x['command']}'")
def exec_on_tpu(host: str, zone: str, command: str, worker: SliceIndex = 0) -> str:
    cmd = TPU_CMD.split(' ') + ["ssh", f"ubuntu@{host}", f"--zone", zone, "--command", command, "--worker", str(worker)]
    return retry_delete(host, zone, cmd, 2)


def all_tpus(zone: str):
    zone = 'projects/' + PROJECT + '/locations/' + zone
    if GLOBAL_DICT.get(f"last_write_{zone}", 0) < time.time() - CACHE_TIME:
        GLOBAL_DICT[f"last_write_{zone}"] = time.time()
        GLOBAL_DICT[f"tpus_{zone}"] = json.loads(call(f"{TPU_CMD} list --zone {zone} --format json"))
    return GLOBAL_DICT[f"tpus_{zone}"]


def valid_tpu(tpu: dict, preempted: bool = True, deleting: bool = False, unhealthy: bool = True) -> bool:
    state = "state" in tpu and (deleting or tpu['state'] != "DELETING") and (preempted or tpu['state'] != "PREEMPTED")
    state |= deleting and preempted
    healthy = unhealthy or "health" not in tpu or tpu["health"] == "HEALTHY"  # we assume no health info == good
    return state and healthy


def tpu_names(zone: str, preempted: bool = True, deleting: bool = False, unhealthy: bool = False,
              no_filter: bool = False, prefix: str = ''):
    while True:
        try:
            tpus = all_tpus(zone)
            if no_filter:
                tpus = [t['name'].split('/')[-1] for t in tpus]
            else:
                tpus = [t['name'].split('/')[-1] for t in tpus if valid_tpu(t, preempted, deleting, unhealthy)]
            return [t for t in tpus if t.startswith(prefix)]
        except KeyboardInterrupt as exc:
            raise exc
        except:
            pass


def delete_no_check(host: str, zone: str, asynchronous: bool):
    os.system(f"echo y | {TPU_CMD} delete {host} --zone {zone} {'--async' * asynchronous}")


def tpu_ips(host: str, zone: str) -> typing.List[str]:
    out = call(f"{TPU_CMD} describe {host} --format json --zone {zone}")
    return [host["accessConfig"]["externalIp"] for host in json.loads(out)["networkEndpoints"]]


def delete_one_tpu(prefix: str, host: str, zone: str, asynchronous: bool = True):
    if prefix not in host or host not in tpu_names(zone, no_filter=True):
        return
    log(f"\x1b[32;1m  DELETING {host}\x1b[0m", log_level=logging.INFO)
    delete_no_check(host, zone, asynchronous)
    while not asynchronous and host in tpu_names(zone, no_filter=True):
        delete_no_check(host, zone, asynchronous)


def delete_all(prefix: str, zone: str):
    while tpu_names(zone, prefix=prefix, no_filter=True):
        threads = [threading.Thread(target=delete_one_tpu, args=(prefix, host, zone, False), daemon=True) for host in
                   tpu_names(zone, prefix=prefix)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def create_tpu(host: str, zone: str, tpu_version: int, preemptible: bool, service_account: str,
               semaphore: typing.Optional[typing.ContextManager], slices: int = 1):
    with semaphore:
        os.system(f'while ! gcloud alpha compute tpus tpu-vm create {host} --service-account {service_account} '
                  f'--zone {zone} --accelerator-type v{tpu_version}-{slices * 8} --version v2-alpha '
                  f'{"--preemptible" * preemptible}; do echo; done')


def recreate(host: str, zone: str, tpu_version: int, preemptible: bool, service_account: str, slices: int,
             creation_semaphore: typing.Optional[typing.ContextManager] = None):
    delete_one_tpu("", host, zone, False)
    create_tpu(host, zone, tpu_version, preemptible, service_account, creation_semaphore, slices)


def get_name(fn: typing.Callable, base: str):
    if hasattr(fn, '__name__'):
        return f"{base} ({fn.__name__})"
    if not isinstance(fn, types.FunctionType):
        return get_name(type(fn), base)
    return base


def start_single(host: str, tpu_version: int, zone: str, preemptible: bool, service_account: str, slices: int,
                 start_fn: typing.Callable[[Context, SliceIndex], None],
                 creation_callback: typing.Callable[[str, typing.Optional[Context]], Context],
                 creation_semaphore: typing.Optional[typing.ContextManager] = None, all_workers: bool = False,
                 unhealthy_timeout_seconds=3600):
    if creation_semaphore is None:
        creation_semaphore = nullcontext()

    ctx = None

    creation_callback_name = get_name(creation_callback, "creation_callback")
    start_fn_name = get_name(start_fn, "start_fn")
    unhealthy_timeout_seconds /= CACHE_TIME

    while True:
        try:
            log("Recreating TPU", log_level=logging.INFO)
            recreate(host, zone, tpu_version, preemptible, service_account, slices, creation_semaphore)
            log(f"TPU Created. Calling {creation_callback_name}.", log_level=logging.INFO)
            ctx = creation_callback(host, ctx)
            log(f"Callback returned. Launching {start_fn_name}", log_level=logging.INFO)
            if all_workers:
                threads = [multiprocessing.Process(target=start_fn, args=(ctx, "all"), daemon=True)]
            else:
                threads = [multiprocessing.Process(target=start_fn, args=(ctx, i), daemon=True) for i in range(slices)]
            for t in threads:
                t.start()
            log("Started start_fn. Babysitting TPU..", log_level=logging.INFO)
            retries = unhealthy_timeout_seconds  # sometimes "unhealthy" resolves itself. Let's wait
            while host in tpu_names(zone, preempted=False, unhealthy=True):
                if retries <= 0:
                    break
                time.sleep(CACHE_TIME)
                if host in tpu_names(zone, preempted=False, unhealthy=False):
                    retries = unhealthy_timeout_seconds
                else:
                    retries -= 1
            log(f"TPU is {'unhealthy' if retries <= 0 else 'preempted'}. Recreating it now.", log_level=logging.INFO)
            while any(t.is_alive() for t in threads):
                for t in threads:
                    if t.is_alive():
                        os.kill(t.pid, signal.SIGINT)
                time.sleep(0.1)
            log("Sent SIGINT to all workers", log_level=logging.INFO)
        except KeyboardInterrupt:
            log(f"{host} - {datetime.datetime.now()}: KeyboardInterrupt received. Killing TPU, then self.",
                log_level=logging.WARN)
            delete_one_tpu("", host, zone, False)
            return


def start_multiple(prefix: str, tpu_version: int, zone: str, preemptible: bool, service_account: str, slices: int,
                   start_fn: typing.Callable[[typing.Any, int], None],
                   created_callback: typing.Callable[[typing.Any], typing.Any], tpus: int):
    procs = []
    creation_semaphore = threading.Semaphore(2)
    for tpu_id in range(tpus):
        proc = threading.Thread(target=start_single, daemon=True, args=(
            f'{prefix}-{tpu_id}', tpu_version, zone, preemptible, service_account, slices, start_fn, created_callback,
            creation_semaphore))
        proc.start()
        procs.append(proc)
    while all(t.is_alive() for t in procs):
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            log(f"MAIN - {datetime.datetime.now()}: KeyboardInterrupt received. Killing All TPUs, then self.",
                logging.WARN)
            delete_all(prefix, zone)
            return
