import { describe, expect, it } from "vitest";

import { Dataset } from "../core/dataset";
import { KinematicsSeries, StepFlag } from "../core/kinematics";
import { buildFixture, straightDrive } from "../core/testing";
import { windowFlag } from "./window_scrubber";

function teleportAt(badStep: number) {
  return (step: number, row: Float32Array): void => {
    straightDrive(step, row);
    if (step === badStep) row[9] = 1000; // dpos_x — implausible jump for a 1 s step
  };
}

describe("windowFlag", () => {
  it("flags only the windows that contain the teleporting step", () => {
    // 40 steps, stride 10, window_size 20 → windows [0,20) [10,30) [20,40); teleport at step 15
    // falls inside windows 0 and 1 but not window 2.
    const fx = buildFixture([{ label: "GridSybil", nSteps: 40, sender: 1, fill: teleportAt(15) }]);
    const ds = Dataset.fromParts(fx.manifest, fx.steps, fx.times);
    const identity = ds.identity(0);
    const kinematics = KinematicsSeries.compute(identity);

    expect(windowFlag(identity, kinematics, 0) & StepFlag.Teleport).not.toBe(0);
    expect(windowFlag(identity, kinematics, 1) & StepFlag.Teleport).not.toBe(0);
    expect(windowFlag(identity, kinematics, 2)).toBe(StepFlag.None);
  });

  it("returns None for a clean straight drive", () => {
    const fx = buildFixture([{ label: "Benign", nSteps: 30 }]);
    const ds = Dataset.fromParts(fx.manifest, fx.steps, fx.times);
    const identity = ds.identity(0);
    const kinematics = KinematicsSeries.compute(identity);
    for (let k = 0; k < identity.windowCount; k++) {
      // Step 0 of every identity has dt=0 (UndefinedDt), which is not Teleport/Inconsistent.
      expect(windowFlag(identity, kinematics, k) & (StepFlag.Teleport | StepFlag.Inconsistent)).toBe(0);
    }
  });
});
