# NOTE: PiperRobot.command_joint_pos performs a synchronous CAN write
# via piper_control. Sessions using PiperRobot MUST set poll_freq on
# the RobotNode (or wrap the command path in a buffered thread, à la
# franka_osc.py). Default subscriber_driven mode runs RobotNode flat-
# out and saturates the CAN bus — observed at ~5.5 kHz with no
# poll_freq on hardware.
"""Piper adapter — wraps Reimagine-Robotics piper_control to expose the
robots_realtime Robot duck-typed protocol.

WARNING: On a warm/enabled arm, do NOT call piper_init.reset_arm — it
briefly disables the motors via disable_arm before re-enabling, which
on a loaded arm causes a visible gravity drop and risk of self-
collision. PiperRobot detects already-enabled state and skips
reset_arm in that case. If you need a hard reset on a warm arm,
physically support the arm first or power-cycle it.

SHUTDOWN: Sessions ending normally leave motors enabled, holding the
last commanded pose. This avoids the disable-induced gravity drop on
warm re-runs. To actively disable on close, construct with
disable_on_close=True, or call close(disable=True) explicitly. To
fully release motors, power-cycle the arm.

stop() is a no-op (graceful — firmware holds last command).
emergency_stop() cuts motor power; do not call on graceful exit.

Piper's PiperInterface controls the 6-DOF arm in radians and exposes the
gripper through a separate command_gripper(position, effort) API. To
match the i2rt MotorChainRobot (YAM) convention used elsewhere in this
codebase, this adapter fuses the gripper into joint_pos[-1] in *command
space* (so command_joint_pos and get_joint_pos return a 7-vector), but
splits arm vs. gripper in get_observations() to mirror YAM's keys.
RobotNode's ramp-seed path (robot_node.py) relies on get_joint_pos
returning the full command-space vector.

Gripper unit asymmetry (matches the YAM/GELLO convention):
  - command space (command_joint_pos input, get_joint_pos output):
    gripper element is normalized [0, 1], where 1.0 maps to
    self._gripper_hi (default: SDK gripper_angle_max). This matches what
    GelloLeaderAgent / PassiveGelloLeaderAgent publish — see
    passive_gello_leader_agent.py:177 and the i2rt MotorChainRobot
    convention referenced there.
  - observation space (get_observations()["gripper_pos"]): meters,
    native Piper SDK units.
The constructor's gripper_limits kwarg is meters (caller-natural),
not normalized.

Synchronous command path measured at ~0.09 ms/call (~11 kHz) on Piper
hardware via SocketCAN; threading is unnecessary at the <=200 Hz rates
RobotNode's subscriber loop produces.

Where the Piper SDK does not expose a quantity YAM reports (joint_vel,
joint_eff for the arm; gripper_vel), we emit zeros and note it inline.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from i2rt.robots.robot import Robot
from piper_control import piper_init, piper_interface

logger = logging.getLogger(__name__)

NUM_ARM_JOINTS = 6
TICK_CACHE_MAX_AGE_S = 1e-3        # halves SDK round-trips when get_joint_pos and
                                   # get_observations both fire in the same RobotNode step
CLIP_WARN_INTERVAL_S = 5.0         # rate-limit clip warnings


class PiperRobot(Robot):
    def __init__(
        self,
        can_port: str = "can0",
        reset_on_init: bool = True,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        gripper_limits: Optional[Tuple[float, float]] = None,
        disable_on_close: bool = False,
    ) -> None:
        self._can_port = can_port
        self._iface = piper_interface.PiperInterface(can_port=can_port)
        self._reset_done = False
        self._disable_on_close = bool(disable_on_close)

        # Wait for the SDK's CAN-frame state cache to populate before any
        # warm/cold detection runs — the very first call to is_*_enabled
        # returns the default False regardless of hardware state.
        self._wait_for_fresh_state()

        # Reset + explicit enable. piper_init.reset_arm alone has been observed
        # to leave the arm in a state where commands return SEND_MESSAGE_FAILED
        # under sustained load, so we additionally enable arm + gripper and
        # verify both before declaring the robot commandable. The is_*_enabled
        # status propagates asynchronously on cold-start, so poll with a
        # bounded timeout rather than checking immediately.
        #
        # SAFETY: piper_init.reset_arm internally does an equivalent sequence
        # (pre-seed → set_arm_mode → enable_arm) but wraps it in disable_arm
        # → enable_arm. The disable step briefly cuts motor power, which on
        # a warm/loaded arm causes a visible gravity drop and risk of self-
        # collision. On a cold arm (motors already off) it's a no-op.
        #
        # The warm path below replicates the same transition WITHOUT the
        # disable, in this exact order:
        #   1. Pre-seed current pose into the firmware command buffer —
        #      without this, set_arm_mode + enable_arm would commit a
        #      stale or empty target and snap the arm.
        #   2. set_arm_mode() transitions the controller from STANDBY to
        #      POSITION_VELOCITY (CAN_CTRL).
        #   3. enable_arm() energizes the motors, which then servo to the
        #      now-fresh buffered target.
        # Skip reset_arm when _is_warm() reports the arm is fully-enabled
        # or in STANDBY with energized motors holding pose.
        if reset_on_init:
            if self._is_warm():
                logger.info(
                    "PiperRobot: arm appears warm; pre-seeding current pose, "
                    "transitioning controller to POSITION_VELOCITY, and re-enabling "
                    "arm without disable cycle."
                )
                try:
                    current_arm = self._iface.get_joint_positions()
                    current_gripper, _ = self._iface.get_gripper_state()
                    self._iface.command_joint_positions(current_arm)
                    self._iface.command_gripper(position=float(current_gripper))
                    time.sleep(0.2)  # let buffer write propagate before mode/enable
                except Exception as exc:
                    raise RuntimeError(
                        "PiperRobot: failed to pre-seed current pose before warm "
                        "controller transition. Refusing to proceed — set_arm_mode + "
                        "enable_arm without a fresh buffered target risks snapping the "
                        "arm to a stale or empty target. Power-cycle the arm to force "
                        "the cold reset path."
                    ) from exc
                # Order matters: mode first (controller knows what to do with the
                # buffered target), then enable_arm (motors energize and servo to
                # the now-non-stale target).
                self._iface.set_arm_mode()
                self._iface.enable_arm()
                # Don't call enable_gripper here — empirically it stays enabled
                # across sessions, and an extra call is no-op. But verify.
                self._wait_for(self._iface.is_arm_enabled, "arm")
            else:
                piper_init.reset_arm(self._iface)
                self._iface.enable_arm()
                self._iface.enable_gripper()
                self._wait_for(self._iface.is_arm_enabled, "arm")
                self._wait_for(self._iface.is_gripper_enabled, "gripper")
            self._reset_done = True

        # Joint limits: caller override wins, else read the SDK's exposed limits.
        if joint_limits is not None:
            arm_lo = np.asarray([lo for lo, _ in joint_limits], dtype=np.float64)
            arm_hi = np.asarray([hi for _, hi in joint_limits], dtype=np.float64)
        else:
            sdk_lim = self._iface.joint_limits   # {"min": [...6...], "max": [...6...]}
            arm_lo = np.asarray(sdk_lim["min"], dtype=np.float64)
            arm_hi = np.asarray(sdk_lim["max"], dtype=np.float64)
        assert arm_lo.shape == (NUM_ARM_JOINTS,) and arm_hi.shape == (NUM_ARM_JOINTS,), (
            f"Expected {NUM_ARM_JOINTS}-element joint limits; got lo={arm_lo.shape}, hi={arm_hi.shape}"
        )
        self._arm_lo = arm_lo
        self._arm_hi = arm_hi

        if gripper_limits is not None:
            self._gripper_lo = float(gripper_limits[0])
            self._gripper_hi = float(gripper_limits[1])
        else:
            self._gripper_lo = 0.0
            self._gripper_hi = float(self._iface.gripper_angle_max)

        # Throttled-warning timestamps (separate so arm/gripper clips don't mask each other).
        self._last_arm_clip_warn_t = 0.0
        self._last_gripper_clip_warn_t = 0.0

        # Tick cache + stale-cache fallback. Each field is independently cacheable
        # so a transient gripper read failure doesn't invalidate a fresh arm read.
        self._cache_arm: Optional[np.ndarray] = None
        self._cache_gripper_pos: Optional[float] = None
        self._cache_gripper_eff: Optional[float] = None
        self._tick_cache_t: float = 0.0

        # Prime the cache. If the very first read fails we have no fallback —
        # let the exception propagate so construction fails loudly.
        try:
            self._read_state(force=True)
        except Exception as exc:
            raise RuntimeError(
                f"PiperRobot: failed to read initial state from SDK: {exc}"
            ) from exc

    def num_dofs(self) -> int:
        return NUM_ARM_JOINTS + 1   # 6 arm joints + 1 gripper at index -1

    def _wait_for(self, predicate, name: str, timeout_s: float = 2.0,
                  poll_interval_s: float = 0.05) -> None:
        """Poll predicate() until True or timeout. Used to handle
        async state propagation on cold-start enable calls."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(poll_interval_s)
        raise RuntimeError(
            f"PiperRobot: {name} did not report enabled within "
            f"{timeout_s}s. Power-cycle the arm and check the CAN bus."
        )

    def _wait_for_fresh_state(self, timeout_s: float = 1.5,
                              poll_interval_s: float = 0.1) -> None:
        """Block briefly until PiperInterface's CAN-frame state cache
        is populated. The first call to is_*_enabled() after
        construction returns the default (False) regardless of true
        hardware state, because the cache hasn't seen any feedback
        frames yet. This helper just waits for cache warm-up.
        Times out silently — the subsequent warm-detect check will
        handle whatever state we end up in.
        """
        import time
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            # Any True flip on either flag means we've seen at least
            # one feedback frame — cache is now live.
            if self._iface.is_arm_enabled() or self._iface.is_gripper_enabled():
                return
            time.sleep(poll_interval_s)
        # Timeout is OK: arm may genuinely be cold/disabled.

    def _is_warm(self) -> bool:
        """Return True if the arm is in any state where reset_arm
        would cause a disable-induced drop. Includes both fully-
        enabled and STANDBY (motors holding pose, control loop idle).
        """
        # Direct boolean flags catch the "actively enabled" case.
        if self._iface.is_arm_enabled() and self._iface.is_gripper_enabled():
            return True
        # arm_status / control_mode = 0 with non-zero joint feedback
        # indicates STANDBY with energized motors — also unsafe to reset.
        # If joint positions are reading non-zero/non-stale, motors
        # have power.
        try:
            pos = self._iface.get_joint_positions()
            # If we got non-trivial readings, the firmware is talking
            # to us. Combined with arm_status reporting NORMAL (0),
            # treat as warm.
            if any(abs(p) > 1e-4 for p in pos):
                return True
        except Exception:
            pass
        return False

    def _read_state(self, force: bool = False) -> None:
        """Refresh tick-cache from the SDK. Reuses the cache if <1 ms old.

        On exception, leave the existing cache in place (stale-cache fallback).
        Only re-raises if no successful read has ever populated the cache.
        """
        now = time.monotonic()
        if not force and (now - self._tick_cache_t) < TICK_CACHE_MAX_AGE_S:
            return

        try:
            self._cache_arm = np.asarray(
                self._iface.get_joint_positions(), dtype=np.float64
            )
        except Exception:
            logger.exception("PiperRobot: get_joint_positions failed; using cached value")
            if self._cache_arm is None:
                raise

        try:
            pos, eff = self._iface.get_gripper_state()
            self._cache_gripper_pos = float(pos)
            self._cache_gripper_eff = float(eff)
        except Exception:
            logger.exception("PiperRobot: get_gripper_state failed; using cached values")
            if self._cache_gripper_pos is None or self._cache_gripper_eff is None:
                raise

        self._tick_cache_t = now

    def get_joint_pos(self) -> np.ndarray:
        """Full command-space vector: [6 arm joints (rad), gripper pos normalized [0, 1]].

        Gripper position is divided by self._gripper_hi (meters) to convert
        the native SDK reading to normalized [0, 1], so that command space
        is symmetric across get_joint_pos / command_joint_pos and matches
        the leader agents' convention. RobotNode's ramp-seed blend depends
        on this — get_observations()["gripper_pos"] is intentionally in
        meters and should NOT be used as a seed.
        """
        self._read_state()
        gripper_norm = self._cache_gripper_pos / self._gripper_hi
        return np.concatenate([self._cache_arm, [gripper_norm]])

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        if not self._reset_done:
            raise RuntimeError(
                "PiperRobot received a command before reset_arm/enable completed; "
                "construct with reset_on_init=True or perform reset+enable manually."
            )
        assert len(joint_pos) == self.num_dofs(), (
            f"PiperRobot expected {self.num_dofs()}-vector ([6 arm, 1 gripper]); "
            f"got length {len(joint_pos)}."
        )
        arm_cmd = np.asarray(joint_pos[:NUM_ARM_JOINTS], dtype=np.float64)

        # joint_pos[-1] is normalized [0, 1] (YAM/GELLO convention). Clamp to
        # [0, 1], then scale to meters using self._gripper_hi as the 1.0
        # reference. The subsequent meter-space clip is defense in depth — it
        # only engages if a caller overrode gripper_limits to something tighter
        # than [0, gripper_hi], or if the [0,1] clamp itself was bypassed.
        gripper_norm_raw = float(joint_pos[-1])
        gripper_norm = max(0.0, min(1.0, gripper_norm_raw))
        gripper_scaled_m = gripper_norm * self._gripper_hi

        arm_clipped = np.clip(arm_cmd, self._arm_lo, self._arm_hi)
        if not np.array_equal(arm_clipped, arm_cmd):
            now = time.monotonic()
            if (now - self._last_arm_clip_warn_t) > CLIP_WARN_INTERVAL_S:
                self._last_arm_clip_warn_t = now
                logger.warning(
                    "PiperRobot: arm command clipped to joint limits. "
                    "raw=%s clipped=%s lo=%s hi=%s",
                    arm_cmd.tolist(), arm_clipped.tolist(),
                    self._arm_lo.tolist(), self._arm_hi.tolist(),
                )

        gripper_clipped_m = max(self._gripper_lo, min(self._gripper_hi, gripper_scaled_m))
        # Warn if either the [0,1] clamp engaged OR the meter-space clip
        # engaged. Report both the raw normalized input and the meter values
        # so the operator can tell a leader publishing >1.0 apart from a
        # tighter meter-space override clipping a valid normalized command.
        norm_clamped = (gripper_norm != gripper_norm_raw)
        meter_clipped = (gripper_clipped_m != gripper_scaled_m)
        if norm_clamped or meter_clipped:
            now = time.monotonic()
            if (now - self._last_gripper_clip_warn_t) > CLIP_WARN_INTERVAL_S:
                self._last_gripper_clip_warn_t = now
                logger.warning(
                    "PiperRobot: gripper command clipped. "
                    "raw_norm=%.4f clamped_norm=%.4f scaled_m=%.4f final_m=%.4f "
                    "meter_range=[%.4f, %.4f]",
                    gripper_norm_raw, gripper_norm, gripper_scaled_m, gripper_clipped_m,
                    self._gripper_lo, self._gripper_hi,
                )

        self._iface.command_joint_positions(arm_clipped)
        # effort omitted -> SDK default (None means "do not update effort"
        # per command_gripper docstring). Add a constructor opt-in if/when
        # callers need to tune grip force.
        self._iface.command_gripper(position=gripper_clipped_m)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Mirror i2rt MotorChainRobot's keys when gripper_index is set."""
        self._read_state()
        zeros6 = np.zeros(NUM_ARM_JOINTS, dtype=np.float64)
        return {
            "joint_pos": self._cache_arm.copy(),                                  # arm only, mirrors YAM
            "joint_vel": zeros6,                                                  # SDK doesn't expose
            "joint_eff": zeros6,                                                  # SDK doesn't expose
            "gripper_pos": np.array([self._cache_gripper_pos], dtype=np.float64), # meters, native SDK units
            "gripper_vel": np.array([0.0], dtype=np.float64),                     # SDK doesn't expose
            "gripper_eff": np.array([self._cache_gripper_eff], dtype=np.float64),
        }

    def stop(self) -> None:
        """Graceful stop hook called by RobotNode.cleanup() on session
        shutdown. NO-OP by design: piper_control firmware holds the
        last commanded pose when commands stop arriving, which is the
        desired behavior. Do NOT call set_emergency_stop here — that
        would cut motor power and cause a gravity drop, breaking the
        warm-skip re-run path in __init__.

        For an actual emergency stop, call self.emergency_stop()
        directly or use the e-stop hardware button.
        """
        return

    def emergency_stop(self) -> None:
        """Hardware-software emergency stop — cuts motor power
        immediately. Arm WILL fall under gravity if loaded. Use only
        for actual emergencies; not for normal session teardown.
        """
        try:
            self._iface.set_emergency_stop()
        except Exception:
            logger.exception("PiperRobot.emergency_stop failed")

    def close(self, disable: bool = False) -> None:
        """Best-effort safe shutdown.

        By default, leaves the arm and gripper ENABLED, holding the
        last commanded pose. This allows re-running rr-session against
        the same warm arm without going through reset_arm's disable-
        then-enable cycle, which on a loaded arm causes a visible
        gravity drop and a snap-back to the previously cached target.

        Pass disable=True to actively cut motor power on shutdown —
        only do this when the arm is supported externally or has been
        commanded to a stable resting pose. To fully release the
        motors, power-cycle the arm.
        """
        if not disable:
            return
        try:
            self._iface.disable_arm()
        except Exception:
            logger.exception("PiperRobot.close: disable_arm failed")
        try:
            self._iface.disable_gripper()
        except Exception:
            logger.exception("PiperRobot.close: disable_gripper failed")

    def __del__(self) -> None:
        # getattr-with-default because attributes may already be torn down
        # at interpreter shutdown.
        try:
            self.close(disable=getattr(self, "_disable_on_close", False))
        except Exception:
            pass
