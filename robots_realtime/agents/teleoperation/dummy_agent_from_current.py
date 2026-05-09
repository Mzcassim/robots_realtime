"""DummyAgentFromCurrent — like DummyAgent but seeds init_q from the
first observation received on a configured state_topic. Used for first
real-hardware motion testing where hardcoding init_q in YAML is unsafe
(the arm may not be at the configured pose at session start).

The 7-element base_q is constructed from [arm_joint_pos[0..5],
gripper_pos / gripper_max_m], converting the gripper from observed
meters to the [0, 1] normalized command-space convention used by
PiperRobot. Until the first observation arrives, act() returns {}
(no command). Once seeded, behaves exactly like DummyAgent — random
offsets around base_q with the configured target_std and pose_interval.
"""

from __future__ import annotations

import numpy as np

from robots_realtime.agents.teleoperation.dummy_agent import DummyAgent


class DummyAgentFromCurrent(DummyAgent):
    def __init__(
        self,
        state_key: str = "state",
        gripper_max_m: float = 0.1,
        target_std: float = 0.3,
        pose_interval: float = 1.0,
    ) -> None:
        # Seed parent with a placeholder base_q; act() returns {} until the
        # first observation overwrites it. Pass init_q (not arm) so the
        # parent's "arm in {'left','right'}" validation is skipped.
        super().__init__(
            init_q=[0.0] * 7,
            target_std=target_std,
            pose_interval=pose_interval,
        )
        self._state_key = state_key
        self._gripper_max_m = float(gripper_max_m)
        self._seeded = False

    def act(self, obs: dict) -> dict:
        if not self._seeded:
            state = obs.get(self._state_key)
            if state is None or "joint_pos" not in state or "gripper_pos" not in state:
                return {}                                    # wait for first observation
            arm = np.asarray(state["joint_pos"], dtype=np.float32)
            gripper_m = float(state["gripper_pos"][0])
            gripper_norm = float(np.clip(gripper_m / self._gripper_max_m, 0.0, 1.0))
            self._base_q = np.concatenate([arm, [gripper_norm]]).astype(np.float32)
            # reset() sets _current_q = _base_q.copy() and _next_draw_t =
            # time.time() + pose_interval, so the first random offset draws
            # one pose_interval after seeding — gives the operator a beat
            # to verify the seeded pose before motion starts.
            self.reset()
            self._seeded = True
        return super().act(obs)
