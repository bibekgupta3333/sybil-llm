import type { Identity } from "../core/identity";

/** Every pixel traceable to a row: identity provenance and the current window's X_windows row. */
export class ProvenanceStrip {
  private identity: Identity | null = null;

  constructor(
    private readonly root: HTMLElement,
    private readonly colorFor: (label: string) => string,
    private readonly onSwitchSource?: (identityId: number) => void,
  ) {}

  setIdentity(identity: Identity, sceneSize: number): void {
    this.identity = identity;
    const r = identity.record;
    const rows = r.x_windows_rows;
    const pseud = r.pseudonyms_total == null
      ? `${r.pseudonyms_surviving} identity`
      : `${r.pseudonyms_surviving} of ${r.pseudonyms_total} pseudonyms survived min_seq_len=20`;
    const sourceLine = r.data_source === "prepared"
      ? "prepared data (<code>X_windows.npy</code>)"
      : "raw VeReMi broadcast — <strong>never in training data</strong>";
    const switchAffordance = r.paired_identity_id !== null
      ? `<button class="btn" data-action="switch-source">→ view the ${r.data_source === "prepared" ? "raw VeReMi broadcast" : "prepared ground truth"} for this vehicle</button>`
      : !identity.isAttacker
        ? `<span class="kv">no raw-broadcast divergence exists for benign vehicles — ground truth is all there is</span>`
        : "";
    this.root.innerHTML = `
      <span class="chip" style="--c:${this.colorFor(r.label)}">${r.label}</span>
      <span class="kv"><b>sender_uid</b> <code>${r.sender_uid}</code></span>
      <span class="kv"><b>vehicle</b> <code>${r.sender ?? "?"}</code> · <b>pseudonym</b> <code>${r.sender_pseudo}</code> · ${pseud}</span>
      <span class="kv"><b>group</b> ${r.group} · <b>run</b> <code>${r.subfolder}</code></span>
      <span class="kv"><b>windows</b> ${r.n_windows} · <b>steps</b> ${r.n_steps} · <b>X_windows rows</b> ${rows[0]}–${rows[rows.length - 1]}</span>
      <span class="kv"><b>data source</b> ${sourceLine}</span>
      ${r.tags.length ? `<span class="kv tags">${r.tags.map((t) => `<span class="tag">${t}</span>`).join("")}</span>` : ""}
      ${sceneSize > 1 ? `<span class="kv"><b>scene</b> ${sceneSize} identities, time-aligned</span>` : ""}
      ${switchAffordance}
      <span class="kv live" data-role="live"></span>`;
    if (r.paired_identity_id !== null) {
      const target = r.paired_identity_id;
      this.root.querySelector('[data-action="switch-source"]')?.addEventListener("click", () => this.onSwitchSource?.(target));
    }
  }

  update(step: number, windowIndex: number): void {
    if (!this.identity) return;
    const live = this.root.querySelector<HTMLElement>('[data-role="live"]');
    if (!live) return;
    const row = this.identity.record.x_windows_rows[windowIndex];
    live.innerHTML = `<b>now</b> step ${step} · window ${windowIndex} (X_windows row <code>${row}</code>) · t = ${this.identity.times[step].toFixed(2)} s`;
  }
}
