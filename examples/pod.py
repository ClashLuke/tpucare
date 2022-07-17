import argparse
import dataclasses
import typing
from netrc import netrc

import wandb
import yaml

from tpucare import exec_command, exec_on_tpu, send_to_tpu, start_single, synchronous_deletion

_, _, wandb_key = netrc().authenticators("api.wandb.ai")


@dataclasses.dataclass
class Context:
    retry: int
    zone: str
    host: str
    branch: str
    run_name: str
    data_path: str
    config_path: str


def load_config(ctx: Context):
    with open(ctx.config_path, 'r') as f:
        config = f.read()
    config = yaml.safe_load(config)

    wandb_api = wandb.Api()
    config["training"]["do_checkpoint"] = True
    base_checkpoint_path = config["training"]["checkpoint_path"]

    start_step = 0
    for run in wandb_api.runs(f"{config['wandb']['entity']}/{config['wandb']['project']}"):
        if run.name == config['wandb']['name']:
            start_step = run.summary["_step"]
            break
    start_step -= start_step % config["training"]["checkpoint_interval"]

    config["training"]["start_step"] = start_step
    config["data"]["path"] = ctx.data_path
    config["wandb"]["name"] = f"{ctx.run_name}-{ctx.retry}"
    if ctx.retry > 0:
        config["training"]["checkpoint_load_path"] = config["training"]["checkpoint_path"]
    config["training"]["checkpoint_path"] = f"{base_checkpoint_path}-{ctx.retry}"
    return yaml.dump(config)


def start_fn(ctx: Context, worker: int):
    """
    This function gets executed in threads to start a run on a new TPU. It receives the context object returned by
    `creation_callback` as well as the worker id which corresponds to the slice id this code was executed on in a
    multi-host setup. For single-host setups, such as v3-8s, the "worker" will always be set to 0.
    Ideally, it'd copy necessary files to the TPU and then run those. Here, `exec_command` can be used to create an
    execution command that automatically spawns a `screen` session which persists even when the SSH connection gets cut.
    """
    send_to_tpu(ctx.host, ctx.zone, "config.yaml", load_config(ctx), worker)
    cmd = exec_command(repository="https://github.com/HomebrewNLP/HomebrewNLP-Jax", wandb_key=wandb_key,
                       branch=ctx.branch)
    send_to_tpu(ctx.host, ctx.zone, "setup.sh", cmd, worker)
    exec_on_tpu(ctx.host, ctx.zone, "bash setup.sh", worker)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="Name of the TPU")
    parser.add_argument("--tpu-version", type=int, default=3, help="Which TPU version to create (v2-8 or v3-8)")
    parser.add_argument("--zone", type=str, default="europe-west4-a", help="GCP Zone TPUs get created in")
    parser.add_argument("--data-path", type=str, default="gs://ggpt4/the-char-pile/",
                        help="Where the data is stored. Should be changed to a bucket in the correct region")
    parser.add_argument("--preemptible", default=1, type=int,
                        help="Whether to create preemptible or non-preemptible TPUs")
    parser.add_argument("--service-account", type=str,
                        help="Service account that controls permissions of TPU (for example, to ensure EU TPUs "
                             "won't "
                             "use US data)")
    parser.add_argument("--branch", type=str, default="main", help="Branch on github to use")
    parser.add_argument("--slices", default=1, type=int,
                        help="How many TPU slices each TPU should have (1=>vX-8, 4=>vX-32)")
    parser.add_argument("--run-name", type=str, help="Prefix to use for all runs on WandB")
    parser.add_argument("--config-path", type=str, help="Path to config.yaml")
    parser.add_argument("--cleanup", default=0, type=int,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.cleanup:
        synchronous_deletion("", args.host, args.zone)
        return

    def creation_callback(host: str, ctx: typing.Optional[Context]) -> Context:
        if ctx is None:  # first invocation
            return Context(retry=0, zone=args.zone, host=args.host, branch=args.branch, run_name=args.run_name,
                           data_path=args.data_path, config_path=args.config_path)
        ctx.retry += 1
        return ctx

    return start_single(args.host, args.tpu_version, args.zone, args.preemptible, args.service_account,
                        args.slices, start_fn, creation_callback)


if __name__ == '__main__':
    main()
