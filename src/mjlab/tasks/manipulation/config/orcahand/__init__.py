import gymnasium as gym

gym.register(
    id="Mjlab-Manipulation-Orca-Drumstick",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.orca_env_cfg:OrcaDrumstickEnvCfg",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:OrcaDrumstickPPORunnerCfg",
    },
)

gym.register(
    id="Mjlab-Manipulation-Orca-Drumstick-Play",
    entry_point="mjlab.envs:ManagerBasedRlEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.orca_env_cfg:OrcaDrumstickEnvCfg_PLAY",
        "rl_cfg_entry_point": f"{__name__}.rl_cfg:OrcaDrumstickPPORunnerCfg",
    },
)
