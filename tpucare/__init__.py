import datetime
import json
import os
import subprocess
import tempfile
import threading
import time
import typing
from contextlib import nullcontext


def call(cmd: str) -> str:
    return subprocess.check_output(cmd.split()).rstrip().decode()


PROJECT = call("gcloud config get project")
GLOBAL_DICT = {}
CACHE_TIME = 10


def exec_command(repository: str, wandb_key: typing.Optional[str] = None, branch: str = "main",
                 setup_command: str = "(bash setup.sh; exit 0)", run_command: str = "bash run.sh"):
    path = repository.split('/')[-1]
    if path.endswith('.git'):
        path = path[:-len('.git')]
    script = ["sudo apt --fix-missing --fix-broken install -y git python3 python3-pip",
              f"(rm -rf {path} ; pkill -f python3 ; exit 0)", f"git clone --depth 1 --branch {branch} {repository}",
              f"cd {path}"]
    if wandb_key is not None:
        script.append("python3 -m pip install wandb")
        script.append(f"/home/ubuntu/.local/bin/wandb login {wandb_key}")
    script.extend([setup_command, f'screen -dmS model bash -c "cd {path} ; {run_command}"'])
    return ' && '.join(script)


def send_to_tpu(host: str, zone: str, filename_on_tpu: str, command: str, worker: int = 0):
    with tempfile.NamedTemporaryFile(mode='w+') as f:
        f.write(command)
        f.flush()
        os.system(f"gcloud alpha compute tpus tpu-vm scp {f.name} ubuntu@{host}:~/{filename_on_tpu} --zone {zone} "
                  f"--worker {worker}")


def exec_on_tpu(host: str, zone: str, command: str, worker: int = 0):
    print(f"running '{command}' ...", end='')
    start_time = time.time()
    ret = subprocess.call(
            ["gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", f"ubuntu@{host}", f"--zone", zone, "--command",
             command, "--worker", str(worker)])
    if not ret:
        print(f"done after {time.time() - start_time:.1f}s")
        return

    delete_one_tpu(host, host, zone)


def all_tpus(zone: str):
    zone = 'projects/' + PROJECT + '/locations/' + zone
    if GLOBAL_DICT.get(f"last_write_{zone}", 0) < time.time() - CACHE_TIME:
        GLOBAL_DICT[f"last_write_{zone}"] = time.time()
        GLOBAL_DICT[f"tpus_{zone}"] = json.loads(call(f"gcloud compute tpus list --zone {zone} --format json"))
    return GLOBAL_DICT[f"tpus_{zone}"]


def tpu_names(zone: str, preempted: bool = True, deleting: bool = False, prefix: str = ''):
    while True:
        try:
            tpus = all_tpus(zone)
            tpus = [t['name'].split('/')[-1] for t in tpus if
                    "state" in t and (deleting or t['state'] != "DELETING") and (
                            preempted or t['state'] != "PREEMPTED")]
            return [t for t in tpus if t.startswith(prefix)]
        except KeyboardInterrupt as exc:
            raise exc
        except:
            pass


def delete_one_tpu(prefix: str, host: str, zone: str):
    if prefix not in host:
        return
    print(f"\x1b[32;1m  DELETING {host}\x1b[0m")
    os.system(f"echo y | gcloud alpha compute tpus tpu-vm delete {host} --zone {zone} --async")


def synchronous_deletion(prefix: str, host: str, zone: str):
    if prefix not in host:
        return
    while host in tpu_names(zone, deleting=True):
        if host in tpu_names(zone):
            delete_one_tpu(prefix, host, zone)
        time.sleep(CACHE_TIME)


def delete_all(prefix: str, zone: str):
    while tpu_names(zone, prefix=prefix):
        threads = [threading.Thread(target=synchronous_deletion, args=(prefix, host, zone), daemon=True) for host in
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
    if host in tpu_names(zone, preempted=True, deleting=True):
        if host not in tpu_names(zone, preempted=False, deleting=False):
            synchronous_deletion("", host, zone)
            create_tpu(host, zone, tpu_version, preemptible, service_account, creation_semaphore, slices)
    else:
        create_tpu(host, zone, tpu_version, preemptible, service_account, creation_semaphore, slices)


def start_single(host: str, tpu_version: int, zone: str, preemptible: bool, service_account: str, slices: int,
                 start_fn: typing.Callable[[typing.Any, int], None],
                 creation_callback: typing.Callable[[str, typing.Any], typing.Any],
                 creation_semaphore: typing.Optional[typing.ContextManager] = None):
    if creation_semaphore is None:
        creation_semaphore = nullcontext()

    ctx = None
    while True:
        try:
            recreate(host, zone, tpu_version, preemptible, service_account, slices, creation_semaphore)
            ctx = creation_callback(host, ctx)
            threads = [threading.Thread(target=start_fn, args=(ctx, i)) for i in range(slices)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            while host in tpu_names(zone, preempted=False):
                time.sleep(60)

        except KeyboardInterrupt:
            print(f"{host} - {datetime.datetime.now()}: KeyboardInterrupt received. Killing TPU, then self.")
            synchronous_deletion("", host, zone)
            return


def start_multiple(prefix: str, tpu_version: int, zone: str, preemptible: bool, service_account: str, slices: int,
                   start_fn: typing.Callable[[typing.Any, int], None],
                   created_callback: typing.Callable[[typing.Any], typing.Any], tpus: int):
    procs = []
    creation_semaphore = threading.Semaphore(2)
    for tpu_id in range(tpus):
        proc = threading.Thread(target=start_single, daemon=True, args=(
                f'{prefix}-{tpu_id}', tpu_version, zone, preemptible, service_account, slices, start_fn,
                created_callback, creation_semaphore))
        proc.start()
        procs.append(proc)
    while all(t.is_alive() for t in procs):
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print(f"MAIN - {datetime.datetime.now()}: KeyboardInterrupt received. Killing All TPUs, then self.")
            delete_all(prefix, zone)
            return
