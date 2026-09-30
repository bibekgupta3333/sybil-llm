import type { Identity } from "../core/identity";
import { INCONSISTENT_RESIDUAL_MPS, KinematicsSeries, StepFlag, TELEPORT_SPEED_MPS } from "../core/kinematics";

/** Live physics-consistency readout: the label-free signal a pretrained detector can exploit. */
export class ConsistencyPanel {
  private identity: Identity | null = null;
  private kinematics: KinematicsSeries | null = null;
  private lastStep = -1;

  constructor(private readonly root: HTMLElement) {}

  setIdentity(identity: Identity): void {
    this.identity = identity;
    this.kinematics = KinematicsSeries.compute(identity);
    this.lastStep = -1;
    this.root.innerHTML = `
      <div class="gauges">
        ${gauge("implied", "implied speed ‖Δpos‖/dt", "m/s")}
        ${gauge("claimed", "claimed speed ‖spd‖", "m/s")}
        ${gauge("residual", "residual ‖Δpos/dt − spd‖", "m/s")}
        ${gauge("angle", "heading vs velocity", "°")}
      </div>
      <div class="verdict" data-role="verdict"></div>
      <div class="summary" data-role="summary"></div>`;
    const k = this.kinematics;
    const summary = this.root.querySelector<HTMLElement>('[data-role="summary"]')!;
    summary.innerHTML = `identity-level · teleport steps <b>${pct(k.fraction(StepFlag.Teleport))}</b> · inconsistent steps <b>${pct(k.fraction(StepFlag.Inconsistent))}</b> · max implied speed <b>${fmt(k.max(k.impliedSpeed))} m/s</b>`;
  }

  update(step: number): void {
    if (!this.identity || !this.kinematics || step === this.lastStep) return;
    this.lastStep = step;
    const k = this.kinematics;
    this.setGauge("implied", k.impliedSpeed[step], TELEPORT_SPEED_MPS, k.has(step, StepFlag.Teleport));
    this.setGauge("claimed", k.speed[step], 30, false);
    this.setGauge("residual", k.residual[step], INCONSISTENT_RESIDUAL_MPS * 2, k.has(step, StepFlag.Inconsistent));
    const deg = (k.headingAngle[step] * 180) / Math.PI;
    this.setGauge("angle", Number.isFinite(deg) ? Math.abs(deg) : NaN, 180, Math.abs(deg) > 45);

    const verdict = this.root.querySelector<HTMLElement>('[data-role="verdict"]')!;
    if (k.has(step, StepFlag.UndefinedDt)) {
      verdict.className = "verdict neutral";
      verdict.textContent = "dt = 0 — no previous step inside this identity; consistency undefined here";
    } else if (k.has(step, StepFlag.Teleport)) {
      verdict.className = "verdict bad";
      verdict.textContent = `TELEPORT — ${fmt(Math.hypot(this.identity.dposX(step), this.identity.dposY(step)))} m of displacement in ${this.identity.dt(step).toFixed(3)} s (${fmt(k.impliedSpeed[step])} m/s implied)`;
    } else if (k.has(step, StepFlag.Inconsistent)) {
      verdict.className = "verdict warn";
      verdict.textContent = `INCONSISTENT — displacement disagrees with the claimed velocity by ${fmt(k.residual[step])} m/s`;
    } else {
      verdict.className = "verdict good";
      verdict.textContent = "PLAUSIBLE — displacement, speed and heading agree";
    }
  }

  private setGauge(id: string, value: number, scale: number, alarm: boolean): void {
    const el = this.root.querySelector<HTMLElement>(`[data-gauge="${id}"]`);
    if (!el) return;
    const bar = el.querySelector<HTMLElement>(".bar > span")!;
    const val = el.querySelector<HTMLElement>(".val")!;
    if (!Number.isFinite(value)) {
      bar.style.width = "0%";
      val.textContent = "—";
      el.classList.remove("alarm");
      return;
    }
    bar.style.width = `${Math.min(100, (value / scale) * 100)}%`;
    val.textContent = fmt(value);
    el.classList.toggle("alarm", alarm);
  }
}

function gauge(id: string, label: string, unit: string): string {
  return `<div class="gauge" data-gauge="${id}"><div class="lbl">${label} <span class="unit">${unit}</span></div><div class="bar"><span></span></div><div class="val">—</div></div>`;
}

function fmt(v: number): string {
  return Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(1);
}

function pct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}
