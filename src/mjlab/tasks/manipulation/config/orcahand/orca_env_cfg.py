from dataclasses import dataclass

from mjlab.asset_zoo.robots.orca_hand.orca_hand_right_config import (
    ORCA_HAND_RIGHT_ACTION_SCALE,
    ORCA_HAND_RIGHT_CFG,
)
from mjlab.tasks.manipulation.drumstick_env_cfg import (
    DrumstickEnvCfg,
)


@dataclass
class OrcaDrumstickEnvCfg(DrumstickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # foot_contact_sensors = [
        #     ContactSensorCfg(
        #         name=f"{side}_foot_ground_contact",
        #         body1=f"{side}_ankle_roll_link",
        #         body2="terrain",
        #         num=1,
        #         data=("found",),
        #         reduce="netforce",
        #     )
        #     for side in ["left", "right"]
        # ]
        # g1_cfg = replace(ORCA_HAND_RIGHT_CFG, sensors=tuple(foot_contact_sensors))
        self.scene.entities["robot"] = ORCA_HAND_RIGHT_CFG

        # sensor_names = ["left_foot_ground_contact", "right_foot_ground_contact"]
        # geom_names = []
        # for i in range(1, 8):
        #     geom_names.append(f"left_foot{i}_collision")
        # for i in range(1, 8):
        #     geom_names.append(f"right_foot{i}_collision")

        # self.events.foot_friction.params["asset_cfg"].geom_names = geom_names

        self.actions.joint_pos.scale = ORCA_HAND_RIGHT_ACTION_SCALE

        # self.rewards.air_time.params["sensor_names"] = sensor_names
        # self.rewards.pose.params["std"] = {
        #   r"^(left|right)_knee_joint$": 0.6,
        #   r"^(left|right)_hip_pitch_joint$": 0.6,
        #   r"^(left|right)_elbow_joint$": 0.6,
        #   r"^(left|right)_shoulder_pitch_joint$": 0.6,
        #   r"^(?!.*(knee_joint|hip_pitch|elbow_joint|shoulder_pitch)).*$": 0.3,
        # }
        # self.rewards.pose.params["std"] = {
        #     # Lower body.
        #     r".*hip_pitch.*": 0.3,
        #     r".*hip_roll.*": 0.15,
        #     r".*hip_yaw.*": 0.15,
        #     r".*knee.*": 0.35,
        #     r".*ankle_pitch.*": 0.25,
        #     r".*ankle_roll.*": 0.1,
        #     # Waist.
        #     r".*waist_yaw.*": 0.15,
        #     r".*waist_roll.*": 0.08,
        #     r".*waist_pitch.*": 0.1,
        #     # Arms.
        #     r".*shoulder_pitch.*": 0.35,
        #     r".*shoulder_roll.*": 0.15,
        #     r".*shoulder_yaw.*": 0.1,
        #     r".*elbow.*": 0.25,
        #     r".*wrist.*": 0.3,
        # }

        self.viewer.body_name = "right_tower"


@dataclass
class OrcaDrumstickEnvCfg_PLAY(OrcaDrumstickEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Effectively infinite episode length.
        self.episode_length_s = int(1e9)
        self.scene.num_envs = 16
