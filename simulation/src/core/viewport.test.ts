import { describe, expect, it } from "vitest";

import { Viewport } from "./viewport";

describe("Viewport", () => {
  it("fits a wide box aspect-locked and centred, flipping y", () => {
    const vp = new Viewport();
    vp.resize(1000, 500);
    vp.fit({ minX: 0, minY: 0, maxX: 2000, maxY: 500 }, { padding: 0 });
    expect(vp.pixelsPerMetre).toBeCloseTo(0.5);
    expect(vp.toCanvasX(1000)).toBeCloseTo(500);
    expect(vp.toCanvasY(250)).toBeCloseTo(250);
    expect(vp.toCanvasY(500)).toBeCloseTo(125); // top of the box sits above centre (y flipped)
    expect(vp.toWorldX(vp.toCanvasX(1234))).toBeCloseTo(1234);
    expect(vp.toWorldY(vp.toCanvasY(77))).toBeCloseTo(77);
  });

  it("enforces a minimum extent for a stationary vehicle", () => {
    const vp = new Viewport();
    vp.resize(600, 600);
    vp.fit({ minX: 10, minY: 10, maxX: 10, maxY: 10 }, { padding: 0, minExtent: 300 });
    expect(vp.pixelsPerMetre).toBeCloseTo(2);
    expect(vp.contains(10, 10)).toBe(true);
    expect(vp.contains(10 + 200, 10)).toBe(false);
  });

  it("refits after a resize and unions boxes", () => {
    const vp = new Viewport();
    vp.resize(400, 400);
    vp.fit({ minX: 0, minY: 0, maxX: 100, maxY: 100 }, { padding: 0, minExtent: 0 });
    vp.resize(800, 800);
    expect(vp.pixelsPerMetre).toBeCloseTo(8);
    const u = Viewport.union({ minX: 0, minY: 5, maxX: 1, maxY: 6 }, { minX: -2, minY: 7, maxX: 0, maxY: 8 });
    expect(u).toEqual({ minX: -2, minY: 5, maxX: 1, maxY: 8 });
  });

  it("zooms around a fixed canvas point, keeping the world point under it", () => {
    const vp = new Viewport();
    vp.resize(400, 400);
    vp.fit({ minX: 0, minY: 0, maxX: 400, maxY: 400 }, { padding: 0, minExtent: 0 });
    const pt = { x: 100, y: 300 };
    const worldBefore = { x: vp.toWorldX(pt.x), y: vp.toWorldY(pt.y) };
    vp.zoomBy(2, pt);
    expect(vp.pixelsPerMetre).toBeCloseTo(2);
    expect(vp.toWorldX(pt.x)).toBeCloseTo(worldBefore.x);
    expect(vp.toWorldY(pt.y)).toBeCloseTo(worldBefore.y);
    expect(vp.isManuallyAdjusted).toBe(true);
  });

  it("pans by a canvas-pixel delta consistently in world space (drag-to-pan: content follows the cursor)", () => {
    const vp = new Viewport();
    vp.resize(400, 400);
    vp.fit({ minX: 0, minY: 0, maxX: 400, maxY: 400 }, { padding: 0, minExtent: 0 });
    const before = vp.toWorldX(200);
    vp.panBy(50, 0);
    // Dragging +50px right moves the map right under the cursor: what used to be at canvas x=150
    // is now at canvas x=200.
    expect(vp.toWorldX(200)).toBeCloseTo(before - 50 / vp.pixelsPerMetre);
  });

  it("reset() restores the last fit and clears the manual flag", () => {
    const vp = new Viewport();
    vp.resize(400, 400);
    vp.fit({ minX: 0, minY: 0, maxX: 400, maxY: 400 }, { padding: 0, minExtent: 0 });
    const scaleBefore = vp.pixelsPerMetre;
    vp.zoomBy(3);
    vp.panBy(20, 20);
    expect(vp.isManuallyAdjusted).toBe(true);
    vp.reset();
    expect(vp.isManuallyAdjusted).toBe(false);
    expect(vp.pixelsPerMetre).toBeCloseTo(scaleBefore);
  });
});
