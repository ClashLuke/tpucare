import argparse
import dataclasses
import typing
from netrc import netrc

import wandb
import yaml

from tputils import delete_all, exec_command, exec_on_tpu, send_to_tpu, start_multiple

_, _, wandb_key = netrc().authenticators("api.wandb.ai")


@dataclasses.dataclass
class Context:
    zone: str
    host: str
    sweep_id: str


def start_fn(ctx: Context, worker: int):
    cmd = exec_command(repository="https://github.com/HomebrewNLP/HomebrewNLP-Jax", wandb_key=wandb_key,
                       run_command=f"/home/ubuntu/.local/bin/wandb agent {ctx.sweep_id}")
    send_to_tpu(ctx.zone, ctx.host, "setup.sh", cmd, worker)
    exec_on_tpu(ctx.zone, ctx.host, "bash setup.sh", worker)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, help="Prefix used to identify TPUs")
    parser.add_argument("--tpu-version", type=int, default=3, help="Which TPU version to create (v2-8 or v3-8)")
    parser.add_argument("--zone", type=str, default="europe-west4-a", help="GCP Zone TPUs get created in")
    parser.add_argument("--preemptible", default=1, type=int,
                        help="Whether to create preemptible or non-preemptible TPUs")
    parser.add_argument("--service-account", type=str,
                        help="Service account that controls permissions of TPU (for example, to ensure EU TPUs "
                             "won't use US data)")
    parser.add_argument("--branch", type=str, default="main", help="Branch on github to use")
    parser.add_argument("--slices", default=1, type=int,
                        help="How many TPU slices each TPU should have (1=>vX-8, 4=>vX-32)")
    parser.add_argument("--config-path", type=str, help="Path to sweep's config.yaml")
    parser.add_argument("--cleanup", default=0, type=int,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.cleanup:
        return delete_all(args.prefix, args.zone)

    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f.read())
    sweep_id = wandb.sweep(config, entity="homebrewnlp", project="gpt")

    def creation_callback(host: str, ctx: typing.Optional[Context]) -> Context:
        if ctx is None:
            return Context(zone=args.zone, host=host, sweep_id=sweep_id)
        return ctx

    return start_multiple(args.host, args.tpu_version, args.zone, args.preemptible, args.service_account,
                          args.slices, start_fn, creation_callback, args.tpus)


if __name__ == '__main__':
    main()
