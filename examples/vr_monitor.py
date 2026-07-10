"""Watch the VR teleop command pipeline -- and check it flows without delay.

Runs the real ``VRTeleop`` control loop and taps it every tick (via the loop's
``on_tick`` hook, so it observes the actual command path, not a copy). Once a
second it reports where any delay would sit:

  * input fps + "fresh N/M ticks" -- of M control ticks this second, how many
    consumed a *new* VR frame. If your headset streams faster than the 50 Hz
    loop, almost every tick is fresh.
  * age (recv->loop) -- how old the VR frame is when the loop acts on it: the
    delay between the server receiving/parsing a frame and the control loop
    consuming it. Should sit well under one 20 ms tick; spikes mean the frames
    arrived in a burst after a gap (WiFi jitter), not that the loop is slow.
  * EE lag -- distance between the *commanded* EE pose and where the arm
    actually is, while the clutch is engaged: the IK + position-servo tracking
    delay, in mm. Small and steady = the arm is riding the command closely.

So "commands arrive without delay" means: age stays low (loop consumes frames
promptly) and EE lag stays small (the arm tracks the command tightly).

Connect a Meta Quest, or in another terminal:  python -m teleop.mock_vr_client

Usage:
  python examples/vr_monitor.py [--no-view] [--port 8081] [--trace]
    --trace  print one line per *new* VR frame (its age + EE), not just 1/sec.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from teleop.vr_teleop import VRTeleop


class Monitor:
    """Per-tick tap that accumulates latency/throughput and prints once a sec."""

    def __init__(self, trace=False):
        self.trace = trace
        self.last_frames = 0
        self.t_last = time.perf_counter()
        self._reset()

    def _reset(self):
        self.ages = []    # recv->loop age (ms) of each fresh frame this window
        self.lags = []    # commanded-vs-actual EE distance (mm) while engaged
        self.frames = 0   # actual VR frames received this window (true rate)
        self.fresh = 0    # ticks that saw >=1 new frame
        self.max_burst = 0  # most frames arriving in a single tick
        self.ticks = 0
        self.engaged = 0

    def on_tick(self, tele):
        now = time.perf_counter()
        snap = tele.state.snapshot()
        ee = tele.data.site_xpos[tele.robot._ee_site]
        self.ticks += 1

        # Frames since the previous tick. >1 means a burst (the server parsed
        # several backed-up frames at once -- the fingerprint of a starved
        # server thread, e.g. GIL held by the render/step-heavy main loop).
        burst = snap.frames - self.last_frames if snap.connected else 0
        self.last_frames = snap.frames
        if burst > 0:
            self.frames += burst
            self.fresh += 1
            self.max_burst = max(self.max_burst, burst)
            age_ms = (now - snap.stamp) * 1000.0 if snap.stamp > 0 else 0.0
            self.ages.append(age_ms)
            if self.trace:
                print(f"  frames +{burst:<2d} (#{snap.frames})  age {age_ms:5.1f}ms  "
                      f"grip {snap.grip:.2f} trig {snap.trigger:.2f}  "
                      f"EE ({ee[0]*100:+.1f},{ee[1]*100:+.1f},{ee[2]*100:+.1f})cm",
                      flush=True)

        if tele._engaged:
            self.engaged += 1
            self.lags.append(float(np.linalg.norm(tele._cmd_pos - ee)) * 1000.0)

        if now - self.t_last >= 1.0:
            self._report(now - self.t_last, snap)
            self.t_last = now
            self._reset()

    def _report(self, dt, snap):
        if not snap.connected:
            print("[monitor] waiting for a VR client...", flush=True)
            return
        true_fps = self.frames / dt   # actual frames/sec (counts bursts)
        fresh_fps = self.fresh / dt   # ticks/sec that got any new data
        am, ax = (np.mean(self.ages), np.max(self.ages)) if self.ages else (0.0, 0.0)
        lm, lx = (np.mean(self.lags), np.max(self.lags)) if self.lags else (0.0, 0.0)
        eng = 100.0 * self.engaged / max(self.ticks, 1)
        # If true_fps stays high but fresh is low with burst>1, frames are
        # arriving clumped (server starved); if true_fps itself drops, fewer
        # frames are actually being received (network / headset side).
        print(f"[monitor] in {true_fps:4.0f}fps (fresh {self.fresh:2d}/{self.ticks:2d}tk "
              f"burst<= {self.max_burst}) | age {am:4.1f}/{ax:4.1f}ms | "
              f"EE lag {lm:5.1f}/{lx:5.1f}mm | eng {eng:3.0f}%",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description="VR teleop command / latency monitor")
    ap.add_argument("--task", default="empty")
    ap.add_argument("--hand", default="right", choices=["right", "left"])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--smooth-tau", type=float, default=0.0,
                    help="command low-pass (s); 0 = off (measure raw latency)")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--trace", action="store_true",
                    help="print one line per new VR frame (not just 1/sec)")
    args = ap.parse_args()

    tele = VRTeleop(task=args.task, hand=args.hand, host=args.host, port=args.port,
                    smooth_tau=args.smooth_tau, view=not args.no_view,
                    show_stats=False)  # the monitor owns the console output
    mon = Monitor(trace=args.trace)
    print("[monitor] tapping the VR control loop. age = recv->loop delay, "
          "EE lag = commanded-vs-actual. Hold grip to engage. Ctrl-C to stop.")
    tele.run(on_tick=mon.on_tick)


if __name__ == "__main__":
    main()
