"""Script to spin up environment without an agent."""

import sys
from dataclasses import dataclass
from typing import Any, Literal

import gymnasium as gym
import torch
import tyro
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.third_party.isaaclab.isaaclab_tasks.utils.parse_cfg import (
    load_cfg_from_registry,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserViewer
from mjlab.viewer.base import PolicyProtocol
from typing_extensions import assert_never

from mjlab.tasks import *  # noqa: F403


@dataclass(frozen=True)
class DummyAgentConfig:
    env: Any
    device: str = "cuda:0"
    viewer: Literal["native", "viser"] = "viser"
    render_all_envs: bool = False


class DummyPolicy(PolicyProtocol):
    def __init__(self, env: gym.Env, mode: Literal["zero", "random"] = "random"):
        self.action_space = env.action_space
        self.mode = mode

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros(self.action_space.shape, device=obs.device)
        else:
            return torch.randn(self.action_space.shape, device=obs.device)


def run_zero_agent(task: str, cfg: DummyAgentConfig) -> None:
    configure_torch_backends()

    env = gym.make(
        task,
        cfg=cfg.env,
        device=cfg.device,
        render_mode=None,
    )

    env = RslRlVecEnvWrapper(env, clip_actions=None)

    policy = DummyPolicy(env)

    if cfg.viewer == "native":
        NativeMujocoViewer(env, policy, render_all_envs=cfg.render_all_envs).run()
    elif cfg.viewer == "viser":
        ViserViewer(env, policy, render_all_envs=cfg.render_all_envs).run()
    else:
        assert_never(cfg.viewer)

    env.close()


def main():
    # Parse first argument to choose the task.
    task_prefix = "Mjlab-"
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(
            [k for k in gym.registry.keys() if k.startswith(task_prefix)]
        ),
        add_help=False,
        return_unknown_args=True,
    )
    del task_prefix

    # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
    env_cfg = load_cfg_from_registry(chosen_task, "env_cfg_entry_point")

    args = tyro.cli(
        DummyAgentConfig,
        args=remaining_args,
        default=DummyAgentConfig(env=env_cfg),
        prog=sys.argv[0] + f" {chosen_task}",
        config=(
            tyro.conf.AvoidSubcommands,
            tyro.conf.FlagConversionOff,
        ),
    )
    del env_cfg, remaining_args

    run_zero_agent(chosen_task, args)


if __name__ == "__main__":
    main()