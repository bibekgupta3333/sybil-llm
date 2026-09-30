import type { Identity } from "../core/identity";
import { KinematicsSeries, StepFlag } from "../core/kinematics";

/** Worst kinematic flag across the steps of window `k` — pure, so it's testable without a DOM. */
export function windowFlag(identity: Identity, kinematics: KinematicsSeries, k: number): StepFlag {
  const start = k * identity.stride;
  const end = Math.min(identity.stepCount, start + identity.windowSize);
  let flag = StepFlag.None;
  for (let i = start; i < end; i++) flag |= kinematics.flags[i];
  return flag;
}

/**
 * Direct window-index navigation for the identity the playback clock is following: one tick per
 * stride-10 window, click to jump straight to it. Ticks are tinted by the worst kinematic flag
 * inside that window, so a defect window (teleport/inconsistent) is visible before playing.
 */
export class WindowScrubber {
  private identity: Identity | null = null;
  private kinematics: KinematicsSeries | null = null;
  private current = -1;

  constructor(
    private readonly root: HTMLElement,
    private readonly onSeek: (windowIndex: number) => void,
  ) {}

  setIdentity(identity: Identity): void {
    this.identity = identity;
    this.kinematics = KinematicsSeries.compute(identity);
    this.current = -1;
    this.render();
  }

  setCurrentWindow(k: number): void {
    if (k === this.current) return;
    this.current = k;
    this.render();
  }

  /** Worst flag across the steps of window `k`, for the tick's colour. */
  flagForWindow(k: number): StepFlag {
    if (!this.identity || !this.kinematics) return StepFlag.None;
    return windowFlag(this.identity, this.kinematics, k);
  }

  private render(): void {
    if (!this.identity) {
      this.root.innerHTML = "";
      return;
    }
    const ticks: string[] = [];
    for (let k = 0; k < this.identity.windowCount; k++) {
      const flag = this.flagForWindow(k);
      const cls = flag & StepFlag.Teleport ? "flag-bad" : flag & StepFlag.Inconsistent ? "flag-warn" : "";
      ticks.push(
        `<button class="wtick ${k === this.current ? "on" : ""} ${cls}" data-k="${k}" title="window ${k}${cls ? " (flagged)" : ""}"></button>`,
      );
    }
    this.root.innerHTML = `<div class="wtrack">${ticks.join("")}</div>`;
    this.root.querySelectorAll<HTMLButtonElement>("[data-k]").forEach((b) =>
      b.addEventListener("click", () => this.onSeek(Number(b.dataset.k))));
  }
}
