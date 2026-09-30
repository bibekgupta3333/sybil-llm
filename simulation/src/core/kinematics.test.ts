import { describe, expect, it } from "vitest";

import { Dataset } from "./dataset";
import { INCONSISTENT_RESIDUAL_MPS, KinematicsSeries, StepFlag, TELEPORT_SPEED_MPS } from "./kinematics";
import { buildFixture, straightDrive } from "./testing";

describe("KinematicsSeries", () => {
  it("flags dt <= 0 as undefined rather than producing Infinity", () => {
    const fx = buildFixture([{ label: "Benign", nSteps: 20 }]);
    const k = KinematicsSeries.compute(Dataset.fromParts(fx.manifest, fx.steps, fx.times).identity(0));
    expect(k.has(0, StepFlag.UndefinedDt)).toBe(true);
    expect(Number.isNaN(k.impliedSpeed[0])).toBe(true);
    expect(k.has(1, StepFlag.UndefinedDt)).toBe(false);
    expect(k.impliedSpeed[1]).toBeCloseTo(12);
    expect(k.residual[1]).toBeCloseTo(0);
    expect(k.fraction(StepFlag.Teleport)).toBe(0);
  });

  it("flags a 1 km jump in one second as a teleport and as inconsistent", () => {
    const fx = buildFixture([{
      label: "GridSybil", nSteps: 20,
      fill: (i, row) => {
        straightDrive(i, row);
        if (i === 7) { row[9] = 1000; } // dpos_x = 1000 m with dt = 1 s
      },
    }]);
    const k = KinematicsSeries.compute(Dataset.fromParts(fx.manifest, fx.steps, fx.times).identity(0));
    expect(k.impliedSpeed[7]).toBeGreaterThan(TELEPORT_SPEED_MPS);
    expect(k.has(7, StepFlag.Teleport)).toBe(true);
    expect(k.residual[7]).toBeGreaterThan(INCONSISTENT_RESIDUAL_MPS);
    expect(k.has(7, StepFlag.Inconsistent)).toBe(true);
    expect(k.has(6, StepFlag.Teleport)).toBe(false);
  });

  it("computes a wrap-safe heading-vs-velocity angle", () => {
    expect(KinematicsSeries.signedAngle(1, 0, 1, 0)).toBeCloseTo(0);
    expect(KinematicsSeries.signedAngle(1, 0, 0, 1)).toBeCloseTo(Math.PI / 2);
    expect(Math.abs(KinematicsSeries.signedAngle(1, 0, -1, 1e-6))).toBeCloseTo(Math.PI, 3);
    expect(Math.abs(KinematicsSeries.signedAngle(1, 0, -1, -1e-6))).toBeCloseTo(Math.PI, 3);
    expect(Number.isNaN(KinematicsSeries.signedAngle(1, 0, 0, 0))).toBe(true); // stationary → undefined
  });
});
