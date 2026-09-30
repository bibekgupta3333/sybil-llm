import { describe, expect, it } from "vitest";

import { Playback, type PlaybackClock } from "./playback";

function fakeClock(): PlaybackClock & { flush: (nowMs: number) => void } {
  let pending: ((now: number) => void) | null = null;
  let now = 0;
  return {
    now: () => now,
    requestFrame: (cb) => { pending = cb; return 1; },
    cancelFrame: () => { pending = null; },
    flush: (nowMs) => { now = nowMs; const cb = pending; pending = null; cb?.(nowMs); },
  };
}

describe("Playback", () => {
  const times = new Float64Array([100, 101, 102, 102, 103, 204, 205]); // duplicate at 102, a 101 s gap

  it("finds the step at a time with duplicates resolving to the last one", () => {
    expect(Playback.stepIndexAt(times, 99)).toBe(0);
    expect(Playback.stepIndexAt(times, 101.5)).toBe(1);
    expect(Playback.stepIndexAt(times, 102)).toBe(3);
    expect(Playback.stepIndexAt(times, 150)).toBe(4);
    expect(Playback.stepIndexAt(times, 999)).toBe(6);
  });

  it("compresses long gaps on the virtual timeline only", () => {
    const v = Playback.compressGaps(times, 2);
    expect(Array.from(v)).toEqual([100, 101, 102, 102, 103, 105, 106]);
  });

  it("seeks, clamps at the end, and respects the speed multiplier", () => {
    const clock = fakeClock();
    const pb = new Playback(clock);
    pb.setTimes(times);
    pb.setSpeed(4);
    pb.seekToStep(4);
    expect(pb.stepIndex).toBe(4);
    pb.play();
    clock.flush(0);
    clock.flush(500); // 0.5 s real × 4 = 2 s sim → 105
    expect(pb.time).toBeCloseTo(105);
    clock.flush(100_000);
    expect(pb.time).toBe(205);
    expect(pb.playing).toBe(false); // clamped and auto-paused at the end
    expect(pb.progress).toBe(1);
  });

  it("switching gap compression keeps the current step", () => {
    const pb = new Playback(fakeClock());
    pb.setTimes(times);
    pb.seekToStep(5);
    pb.setCompressGaps(true);
    expect(pb.stepIndex).toBe(5);
    expect(pb.duration).toBe(6);
    expect(pb.simulationTime).toBe(204);
  });

  it("notifies frame listeners on seek", () => {
    const pb = new Playback(fakeClock());
    pb.setTimes(times);
    const seen: number[] = [];
    pb.onFrame((_t, step) => seen.push(step));
    pb.seekToStep(2);
    pb.stepBy(1);
    expect(seen).toEqual([2, 3]);
  });
});
