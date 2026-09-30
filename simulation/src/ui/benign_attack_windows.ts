import type { Dataset } from "../core/dataset";
import type { Identity } from "../core/identity";
import { Playback } from "../core/playback";
import { identityTraceLabel } from "./controls";
import { WindowComparePanel } from "./window_compare_panel";

/**
 * The bottom panel of "Benign vs attack": one frozen benign window next to one frozen attack window,
 * picked by window index (or jumped to the playhead), on top of the shared `WindowComparePanel`.
 * With several attacks selected, a dropdown picks which attack fills the right column.
 */
export class BenignAttackWindows {
  private readonly panel: WindowComparePanel;
  private readonly caption: HTMLElement;
  private readonly benignTicks: HTMLElement;
  private readonly attackTicks: HTMLElement;
  private readonly attackSelect: HTMLSelectElement;
  private benign: Identity | null = null;
  private attacks: readonly Identity[] = [];
  private attackIndex = 0;
  private benignWindow = 0;
  private attackWindow = 0;

  constructor(root: HTMLElement, dataset: Dataset, private readonly simTime: () => number) {
    root.innerHTML = `
      <div class="panel-head">
        <span class="panel-title">Window compare · benign window vs attack window</span>
        <span class="panel-caption" data-role="caption"></span>
      </div>
      <div class="bwc-controls">
        <div class="bwc-row"><span class="bwc-label bwc-benign">x · benign window</span><div class="wnum-row" data-role="benign-ticks"></div></div>
        <div class="bwc-row"><span class="bwc-label bwc-attack">y · attack</span><select data-role="attack-select"></select>
          <div class="wnum-row" data-role="attack-ticks"></div></div>
        <button class="btn bwc-sync" data-role="sync">⇣ Both windows at the playhead</button>
      </div>
      <div class="wc-grid" data-role="panel"></div>`;
    this.caption = root.querySelector<HTMLElement>('[data-role="caption"]')!;
    this.benignTicks = root.querySelector<HTMLElement>('[data-role="benign-ticks"]')!;
    this.attackTicks = root.querySelector<HTMLElement>('[data-role="attack-ticks"]')!;
    this.attackSelect = root.querySelector<HTMLSelectElement>('[data-role="attack-select"]')!;
    this.panel = new WindowComparePanel(root.querySelector<HTMLElement>('[data-role="panel"]')!, dataset,
      { truth: "x · Benign window", attack: "y · Attack window" });
    this.attackSelect.addEventListener("change", () => {
      this.attackIndex = Number(this.attackSelect.value);
      this.attackWindow = 0;
      this.apply();
    });
    root.querySelector('[data-role="sync"]')!.addEventListener("click", () => this.jumpToPlayhead());
  }

  /** Show `benign` and the given attacks (already clipped to the shared window). */
  setView(benign: Identity, attacks: readonly Identity[]): void {
    const sameBenign = this.benign === benign;
    this.benign = benign;
    this.attacks = attacks;
    if (!sameBenign) this.benignWindow = 0;
    if (this.attackIndex >= attacks.length) this.attackIndex = 0;
    this.attackWindow = Math.min(this.attackWindow, this.currentAttack()!.windowCount - 1);
    this.attackSelect.innerHTML = attacks.map((a, i) =>
      `<option value="${i}" ${i === this.attackIndex ? "selected" : ""}>vehicle ${a.record.sender ?? "?"} · ${identityTraceLabel(a)}</option>`).join("");
    this.attackSelect.disabled = attacks.length < 2;
    this.apply();
  }

  /** Call after the panel becomes visible (its canvases measure 0×0 while hidden). */
  remeasure(): void {
    this.panel.remeasure();
  }

  private currentAttack(): Identity | null {
    return this.attacks[this.attackIndex] ?? null;
  }

  private jumpToPlayhead(): void {
    const attack = this.currentAttack();
    if (!this.benign || !attack) return;
    const t = this.simTime();
    this.benignWindow = this.benign.windowIndexOf(Playback.stepIndexAt(this.benign.times, t));
    this.attackWindow = attack.windowIndexOf(Playback.stepIndexAt(attack.times, t));
    this.apply();
  }

  private apply(): void {
    const attack = this.currentAttack();
    if (!this.benign || !attack) return;
    this.renderTicks(this.benignTicks, this.benign, this.benignWindow, (k) => { this.benignWindow = k; this.apply(); });
    this.renderTicks(this.attackTicks, attack, this.attackWindow, (k) => { this.attackWindow = k; this.apply(); });
    this.panel.setSide("truth", this.benign, this.benignWindow);
    this.panel.setSide("attack", attack, this.attackWindow);
    const tb = this.benign.times[this.benignWindow * this.benign.stride];
    const ta = attack.times[this.attackWindow * attack.stride];
    this.caption.textContent = `benign window ${this.benignWindow} starts t = ${tb.toFixed(1)} s · attack window ${this.attackWindow} starts t = ${ta.toFixed(1)} s`;
  }

  private renderTicks(host: HTMLElement, identity: Identity, current: number, onPick: (k: number) => void): void {
    host.innerHTML = Array.from({ length: identity.windowCount }, (_, k) =>
      `<button class="wnum ${k === current ? "on" : ""}" data-k="${k}">${k}</button>`).join("");
    host.querySelectorAll<HTMLButtonElement>("[data-k]").forEach((b) =>
      b.addEventListener("click", () => onPick(Number(b.dataset.k))));
  }
}
