"""HoldPoseAgent — publishes the robot's currently observed pose as a
command. Used for first-bringup verification of the round-trip command
path on real hardware before introducing motion.

MUST be configured with ``loop_mode: fixed_rate`` (with a bounded
``poll_freq``) when paired with a subscriber_driven RobotNode.
subscriber_driven on both creates an ungoverned feedback loop that
swamps the underlying I/O. See incident: rate runaway at 170 kHz on
Piper, which produced SEND_MESSAGE_FAILED storms on the CAN bus and
caused unintended motion.

Sourcing the target from the robot's own state stream eliminates the
unsafe-init-pose class of bug (e.g., DummyAgent's hardcoded init pose
not matching where the arm physically is at session start). The agent
expects YAM-style observation keys: ``joint_pos`` (arm-only, radians)
and ``gripper_pos`` (meters, 1-vec). The gripper is converted to the
[0, 1] normalized command-space convention using ``gripper_max_m`` as
the 1.0 reference, mirroring PiperRobot's command-vs-observation
gripper unit asymmetry. ``gripper_max_m`` should match the robot's
``_gripper_hi`` (0.1 for default Piper); override if the robot was
constructed with a tighter ``gripper_limits``.
"""

from typing import Any, Dict

import numpy as np


class HoldPoseAgent:
    def __init__(self, state_key: str = "state", gripper_max_m: float = 0.1) -> None:
        self._state_key = state_key
        self._gripper_max_m = float(gripper_max_m)

    def reset(self) -> None:
        return None

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        state = obs.get(self._state_key)
        if state is None or "joint_pos" not in state:
            # No state yet — return empty so AgentNode publishes nothing and
            # RobotNode keeps the motors at their last commanded position.
            return {}

        arm = np.asarray(state["joint_pos"], dtype=np.float32)
        if "gripper_pos" in state:
            gripper_m = float(state["gripper_pos"][0])
            gripper_norm = float(np.clip(gripper_m / self._gripper_max_m, 0.0, 1.0))
        else:
            gripper_norm = 0.0

        full = np.concatenate([arm, [gripper_norm]])
        return {"pos": full}
