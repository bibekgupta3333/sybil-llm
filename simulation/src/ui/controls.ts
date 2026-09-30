import { ATTACK_SIGNATURES, type BenignAttackMatch, type BenignMultiAttackView, type Dataset, type PairedAttacker } from "../core/dataset";
import type { Identity } from "../core/identity";
import type { Scenario } from "../core/scenario";

export type ViewMode = "single" | "sender" | "compare" | "mapCompare" | "benignVsAttack" | "windowCompare";

export interface ControlState {
  readonly mode: ViewMode;
  readonly scenarioKey: string | null;
  readonly primaryId: number;
  readonly compareId: number | null;
  readonly senderKey: string | null;
  readonly pairKey: string | null;
  readonly pairPrimary: "real" | "attack";
  readonly matchKey: string | null;
  readonly matchPrimary: "benign" | "attack";
  /** Attack type shown in Benign vs attack; "all" shows every type, grouped. */
  readonly matchFamily: string;
  /** Extra attack traces (same run, overlapping the match's benign vehicle) shown alongside its attack. */
  readonly extraAttackIds: readonly number[];
  readonly truthId: number | null;
  readonly truthWindow: number;
  readonly attackId: number | null;
  readonly attackWindow: number;
  readonly lockToRoad: boolean;
  readonly compressGaps: boolean;
  readonly normalized: boolean;
}

export interface ControlsHandlers {
  readonly onChange: (state: ControlState) => void;
  readonly onExport: () => void;
  readonly onResetView: () => void;
}

/** Sidebar: scenario picker, filters, identity/window pickers, sender groups, view toggles. */
export class Controls {
  private state: ControlState;
  private labelFilter = "all";
  private groupFilter = "all";
  private tagFilter = "all";
  private readonly senderGroups: Map<string, Identity[]>;

  constructor(
    private readonly root: HTMLElement,
    private readonly dataset: Dataset,
    private readonly handlers: ControlsHandlers,
  ) {
    this.senderGroups = dataset.senderGroups(2);
    const first = dataset.identities().find((i) => i.label === "Benign") ?? dataset.identity(0);
    this.state = {
      mode: "single", scenarioKey: null, primaryId: first.id, compareId: null, senderKey: null,
      pairKey: null, pairPrimary: "real", matchKey: null, matchPrimary: "attack", matchFamily: "all", extraAttackIds: [],
      truthId: null, truthWindow: 0, attackId: null, attackWindow: 0,
      lockToRoad: false, compressGaps: true, normalized: true,
    };
    this.render();
  }

  get current(): ControlState {
    return this.state;
  }

  setPrimary(id: number): void {
    this.update({ primaryId: id });
  }

  /** Jump to a specific identity from anywhere — used by the provenance strip's switch-source button. */
  jumpToIdentity(id: number): void {
    this.update({ mode: "single", primaryId: id });
  }

  toggle(key: "lockToRoad" | "compressGaps" | "normalized"): void {
    this.update({ [key]: !this.state[key] } as Partial<ControlState>);
  }

  activeScenario(): Scenario | null {
    return this.state.scenarioKey ? this.dataset.scenario(this.state.scenarioKey) ?? null : null;
  }

  /** Identities to put on the map for the current state; first is the clock's primary. Empty in windowCompare. */
  scene(): Identity[] {
    const s = this.state;
    if (s.mode === "windowCompare") return [];
    if (s.mode === "mapCompare") {
      const pair = this.pairedCandidates().find((p) => p.key === s.pairKey);
      if (!pair) return [];
      const real = this.dataset.identity(pair.realId);
      const attack = this.dataset.identity(pair.attackId);
      return s.pairPrimary === "attack" ? [attack, real] : [real, attack];
    }
    if (s.mode === "benignVsAttack") {
      const view = this.benignAttackView();
      if (!view) return [];
      const [first, ...rest] = view.attacks;
      return s.matchPrimary === "benign" ? [view.benign, ...view.attacks] : [first, view.benign, ...rest];
    }
    if (s.mode === "sender" && s.senderKey) {
      const group = this.senderGroups.get(s.senderKey);
      if (group && group.length) return group;
    }
    const primary = this.dataset.identity(s.primaryId);
    if (s.mode === "compare" && s.compareId !== null) return [primary, this.dataset.identity(s.compareId)];
    return [primary];
  }

  /** The two frozen (identity, window) picks for Window Compare mode, or null until both are chosen. */
  windowCompareSelection(): { truth: Identity; truthWindow: number; attack: Identity; attackWindow: number } | null {
    const s = this.state;
    if (s.mode !== "windowCompare" || s.truthId === null || s.attackId === null) return null;
    return {
      truth: this.dataset.identity(s.truthId), truthWindow: s.truthWindow,
      attack: this.dataset.identity(s.attackId), attackWindow: s.attackWindow,
    };
  }

  /** The selected benign/attack match (both clipped to their shared time window), if any. */
  benignAttackSelection(): BenignAttackMatch | null {
    const s = this.state;
    if (s.mode !== "benignVsAttack") return null;
    return this.matchCandidates().find((m) => m.key === s.matchKey) ?? null;
  }

  /** The selected match plus any extra attacks, all clipped to one shared window. */
  benignAttackView(): BenignMultiAttackView | null {
    const match = this.benignAttackSelection();
    return match ? this.dataset.benignMultiAttackView(match, this.state.extraAttackIds) : null;
  }

  private update(patch: Partial<ControlState>): void {
    this.state = { ...this.state, ...patch };
    this.render();
    this.handlers.onChange(this.state);
  }

  private inScenario(identity: Identity, scenario: Scenario | null): boolean {
    return !scenario || scenario.benignIds.includes(identity.id) || scenario.attackerIds.includes(identity.id);
  }

  private filtered(): Identity[] {
    const scenario = this.activeScenario();
    return this.dataset.identities().filter((i) =>
      this.inScenario(i, scenario) &&
      (this.labelFilter === "all" || i.label === this.labelFilter) &&
      (this.groupFilter === "all" || i.record.group === this.groupFilter) &&
      (this.tagFilter === "all" || (this.tagFilter === "none" ? i.record.tags.length === 0 : i.hasTag(this.tagFilter))));
  }

  private truthCandidates(): Identity[] {
    const scenario = this.activeScenario();
    return this.dataset.identities().filter((i) => i.label === "Benign" && this.inScenario(i, scenario));
  }

  private attackCandidates(): Identity[] {
    const scenario = this.activeScenario();
    return this.dataset.identities().filter((i) =>
      i.isAttacker && this.inScenario(i, scenario) && (this.labelFilter === "all" || i.label === this.labelFilter));
  }

  /** Benign/attack matches, scoped to the active scenario. */
  private matchCandidates(): BenignAttackMatch[] {
    const scenario = this.activeScenario();
    return this.dataset.benignAttackMatches().filter((m) => !scenario || (m.subfolder === scenario.subfolder && m.group === scenario.group));
  }

  /** Physical vehicles with a resolved prepared/raw-VeReMi pair, scoped to the active scenario. */
  private pairedCandidates(): PairedAttacker[] {
    const scenario = this.activeScenario();
    return this.dataset.pairedAttackers().filter((p) => !scenario || scenario.attackerIds.includes(p.realId));
  }

  private render(): void {
    const labels = [...new Set(this.dataset.records.map((r) => r.label))].sort();
    const s = this.state;
    this.root.innerHTML = `
      <section class="side-block">
        <h2>Mode</h2>
        <div class="seg">
          ${(["single", "sender", "compare", "mapCompare", "benignVsAttack", "windowCompare"] as const).map((m) =>
            `<button data-mode="${m}" class="${s.mode === m ? "on" : ""}">${modeLabel(m)}</button>`).join("")}
        </div>
        <p class="hint">${modeHint(s.mode)}</p>
      </section>

      ${this.renderScenarioPicker()}

      ${s.mode === "sender" ? this.renderSenderGroups()
        : s.mode === "windowCompare" ? this.renderWindowCompare(labels)
        : s.mode === "mapCompare" ? this.renderMapCompare()
        : s.mode === "benignVsAttack" ? this.renderBenignVsAttack()
        : this.renderStandardPickers(labels)}

      <section class="side-block">
        <h2>View</h2>
        <label class="check"><input type="checkbox" data-toggle="lockToRoad" ${s.lockToRoad ? "checked" : ""}> lock map to context extent <kbd>L</kbd></label>
        <label class="check"><input type="checkbox" data-toggle="compressGaps" ${s.compressGaps ? "checked" : ""}> compress gaps &gt; 2 s <kbd>G</kbd></label>
        <label class="check"><input type="checkbox" data-toggle="normalized" ${s.normalized ? "checked" : ""}> heatmap z-scored (model input) <kbd>N</kbd></label>
        <button class="btn" data-action="resetView">⟳ Reset map zoom/pan</button>
        <button class="btn" data-action="export">⤓ Export map frame as PNG</button>
      </section>`;
    this.bind();
  }

  private renderScenarioPicker(): string {
    const scenarios = [...this.dataset.scenarios()].sort((a, b) => b.attackerIds.length - a.attackerIds.length);
    const active = this.state.scenarioKey;
    return `
      <section class="side-block">
        <h2>Scenario (VeReMi run) <span class="count">${scenarios.length}</span></h2>
        <div class="ident-list scenario-list" data-role="scenario">
          <button class="ident ${active === null ? "on" : ""}" data-scenario="">
            <span class="meta">All scenarios</span>
          </button>
          ${scenarios.map((sc) => `<button class="ident ${sc.key === active ? "on" : ""}" data-scenario="${sc.key}">
            <span class="meta">${sc.subfolder} · ${sc.group} · ${sc.benignIds.length} benign / ${sc.attackerIds.length} attacker</span>
          </button>`).join("")}
        </div>
      </section>`;
  }

  private renderStandardPickers(labels: string[]): string {
    const list = this.filtered();
    const s = this.state;
    return `
      <section class="side-block">
        <h2>Filters</h2>
        <label>class <select data-filter="label">${["all", ...labels].map((l) => `<option ${l === this.labelFilter ? "selected" : ""}>${l}</option>`).join("")}</select></label>
        <label>group <select data-filter="group">${["all", "0709", "1416"].map((g) => `<option ${g === this.groupFilter ? "selected" : ""}>${g}</option>`).join("")}</select></label>
        <label>tag <select data-filter="tag">${["all", "none", "stress", "defect_demo"].map((t) => `<option ${t === this.tagFilter ? "selected" : ""}>${t}</option>`).join("")}</select></label>
      </section>
      <section class="side-block grow">
        <h2>${s.mode === "compare" ? "Primary identity" : "Identity"} <span class="count">${list.length}</span></h2>
        <div class="ident-list" data-role="primary">${list.map((i) => this.identRow(i, i.id === s.primaryId)).join("")}</div>
      </section>
      ${s.mode === "compare" ? `
      <section class="side-block grow">
        <h2>Compare with</h2>
        <div class="ident-list" data-role="compare">${list.filter((i) => i.id !== s.primaryId).map((i) => this.identRow(i, i.id === s.compareId)).join("")}</div>
      </section>` : ""}`;
  }

  private renderWindowCompare(labels: string[]): string {
    const s = this.state;
    const truth = this.truthCandidates();
    const attack = this.attackCandidates();
    const truthIdent = s.truthId !== null ? this.dataset.identity(s.truthId) : null;
    const attackIdent = s.attackId !== null ? this.dataset.identity(s.attackId) : null;
    return `
      <section class="side-block">
        <h2>Attack family</h2>
        <label>family <select data-filter="label">${["all", ...labels.filter((l) => l !== "Benign")].map((l) => `<option ${l === this.labelFilter ? "selected" : ""}>${l}</option>`).join("")}</select></label>
      </section>
      <section class="side-block grow">
        <h2>Ground truth window <span class="count">${truth.length}</span></h2>
        <div class="ident-list" data-role="truth">${truth.map((i) => this.identRow(i, i.id === s.truthId)).join("")}</div>
        ${truthIdent ? this.windowTicks("truth", truthIdent, s.truthWindow) : `<p class="hint">Pick a benign identity above.</p>`}
      </section>
      <section class="side-block grow">
        <h2>Attack window <span class="count">${attack.length}</span></h2>
        <div class="ident-list" data-role="attack">${attack.map((i) => this.identRow(i, i.id === s.attackId)).join("")}</div>
        ${attackIdent ? this.windowTicks("attack", attackIdent, s.attackWindow) : `<p class="hint">Pick an attacker identity above.</p>`}
      </section>`;
  }

  private renderMapCompare(): string {
    const s = this.state;
    const pairs = this.pairedCandidates();
    const selected = pairs.find((p) => p.key === s.pairKey) ?? null;
    return `
      <section class="side-block grow">
        <h2>Real + attack pairs <span class="count">${pairs.length}</span></h2>
        <div class="ident-list" data-role="pair">
          ${pairs.map((p) => `<button class="ident ${p.key === s.pairKey ? "on" : ""}" data-pair="${p.key}">
            <span class="chip" style="--c:${this.dataset.colorFor(p.label)}">${p.label}</span>
            <span class="meta">vehicle <code>${p.sender}</code> · ${p.group} · ${p.subfolder}</span>
          </button>`).join("")}
        </div>
        ${pairs.length === 0 ? `<p class="hint">No prepared/raw-VeReMi pairs were exported for this scenario.</p>` : ""}
      </section>
      ${selected ? `
      <section class="side-block">
        <h2>Strips / heatmap / consistency follow</h2>
        <div class="seg">
          <button data-pairprimary="real" class="${s.pairPrimary === "real" ? "on" : ""}">Ground truth</button>
          <button data-pairprimary="attack" class="${s.pairPrimary === "attack" ? "on" : ""}">Attack broadcast</button>
        </div>
      </section>` : ""}`;
  }

  private renderBenignVsAttack(): string {
    const s = this.state;
    const all = this.matchCandidates();
    const families = ATTACK_FAMILIES.map((f) => ({ family: f, count: all.filter((m) => m.label === f).length }));
    const shown = all.filter((m) => s.matchFamily === "all" || m.label === s.matchFamily);
    const selected = all.find((m) => m.key === s.matchKey) ?? null;
    const groups = ATTACK_FAMILIES
      .map((f) => ({ family: f, rows: shown.filter((m) => m.label === f) }))
      .filter((g) => g.rows.length > 0);
    return `
      ${selected ? this.renderComparingCard(selected) : ""}
      <section class="side-block">
        <h2>Attack type <span class="count">${all.length} pairs</span></h2>
        <div class="seg">
          <button data-matchfamily="all" class="${s.matchFamily === "all" ? "on" : ""}">All <span class="n">${all.length}</span></button>
          ${families.map(({ family, count }) => `<button data-matchfamily="${family}" ${count === 0 ? "disabled" : ""}
            class="${s.matchFamily === family ? "on" : ""}" style="--c:${this.dataset.colorFor(family)}">${shortFamily(family)} <span class="n">${count}</span></button>`).join("")}
        </div>
      </section>
      <section class="side-block grow">
        <h2>Benign + attack, same time <span class="count">${shown.length}</span></h2>
        <div class="ident-list" data-role="match">
          ${groups.map((g) => `
            <div class="group-head" style="--c:${this.dataset.colorFor(g.family)}">${g.family} <span class="count">${g.rows.length}</span></div>
            ${g.rows.map((m) => `<button class="ident ${m.key === s.matchKey ? "on" : ""}" data-match="${m.key}">
              <span class="meta"><b>${m.folder}</b><br>${Math.round(m.overlap[1] - m.overlap[0])} s shared ·
              benign vehicle ${m.benign.record.sender ?? "?"} vs attacker ${m.attack.record.sender ?? "?"} · ${traceLabel(m)}</span>
            </button>`).join("")}`).join("")}
        </div>
        ${shown.length === 0 ? `<p class="hint">No benign/attack pairs were exported for this selection.</p>` : ""}
      </section>
      ${selected ? `
      <section class="side-block">
        <h2>Strips / heatmap / consistency follow</h2>
        <div class="seg">
          <button data-matchprimary="benign" class="${s.matchPrimary === "benign" ? "on" : ""}">Benign</button>
          <button data-matchprimary="attack" class="${s.matchPrimary === "attack" ? "on" : ""}">Attack</button>
        </div>
      </section>` : ""}`;
  }

  private renderComparingCard(m: BenignAttackMatch): string {
    const view = this.dataset.benignMultiAttackView(m, this.state.extraAttackIds);
    const others = this.dataset.attacksOverlapping(m.benign.record.id).filter((a) => a.id !== m.attack.id);
    const extras = new Set(this.state.extraAttackIds);
    const [w0, w1] = view.window;
    return `
      <section class="side-block comparing" style="--c:${this.dataset.colorFor(m.label)}">
        <h2>Comparing</h2>
        <div class="cmp-type"><span class="chip" style="--c:${this.dataset.colorFor(m.label)}">${m.label}</span> vs <span class="chip" style="--c:${this.dataset.colorFor("Benign")}">Benign</span></div>
        <p class="cmp-sig">${ATTACK_SIGNATURES[m.label] ?? ""}</p>
        <dl class="cmp-kv">
          <dt>folder</dt><dd><code>${m.folder}</code></dd>
          <dt>benign (x)</dt><dd>vehicle ${m.benign.record.sender ?? "?"} · pseudonym #${m.benign.record.sender_pseudo}</dd>
          <dt>attack (y)</dt><dd>vehicle ${m.attack.record.sender ?? "?"} · ${traceLabel(m)}</dd>
          <dt>window</dt><dd>${Math.round(w1 - w0)} s · t = ${w0.toFixed(0)}–${w1.toFixed(0)}</dd>
        </dl>
        <div class="cmp-more">
          <div class="cmp-more-head">More ${shortFamily(m.label)} attacks against benign vehicle ${m.benign.record.sender ?? "?"}
            <span class="count">${others.length}</span></div>
          ${others.length === 0
            ? `<p class="hint">No other attack trace in this run overlaps this benign vehicle long enough.</p>`
            : others.map((a) => `<label class="check"><input type="checkbox" data-extraattack="${a.id}" ${extras.has(a.id) ? "checked" : ""}>
                vehicle ${a.record.sender ?? "?"} · ${identityTraceLabel(a)}</label>`).join("")}
        </div>
      </section>`;
  }

  private windowTicks(role: "truth" | "attack", identity: Identity, current: number): string {
    const buttons = Array.from({ length: identity.windowCount }, (_, k) =>
      `<button class="wnum ${k === current ? "on" : ""}" data-${role}win="${k}">${k}</button>`).join("");
    return `<div class="wnum-row" role="group" aria-label="window index">${buttons}</div>`;
  }

  private renderSenderGroups(): string {
    const groups = [...this.senderGroups.entries()].sort((a, b) => b[1].length - a[1].length);
    return `
      <section class="side-block grow">
        <h2>Physical senders with ≥2 surviving pseudonyms <span class="count">${groups.length}</span></h2>
        <div class="ident-list" data-role="sender">
          ${groups.map(([key, ids]) => {
            const r = ids[0].record;
            return `<button class="ident ${key === this.state.senderKey ? "on" : ""}" data-sender="${key}">
              <span class="chip" style="--c:${this.dataset.colorFor(r.label)}">${r.label}</span>
              <span class="meta">vehicle <code>${r.sender}</code> · ${ids.length} pseudonyms${r.pseudonyms_total ? ` of ${r.pseudonyms_total}` : ""} · ${r.group}</span>
            </button>`;
          }).join("")}
        </div>
      </section>`;
  }

  private identRow(i: Identity, on: boolean): string {
    const r = i.record;
    return `<button class="ident ${on ? "on" : ""}" data-id="${r.id}">
      <span class="chip" style="--c:${this.dataset.colorFor(r.label)}">${r.label}</span>
      <span class="meta">#${r.id} · ${r.group} · ${r.n_windows} win · ${r.n_steps} steps${r.tags.length ? ` · ${r.tags.join(",")}` : ""}</span>
    </button>`;
  }

  private bind(): void {
    this.root.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((b) =>
      b.addEventListener("click", () => {
        const mode = b.dataset.mode as ViewMode;
        const firstGroup = [...this.senderGroups.keys()][0] ?? null;
        const autoPick = mode === "windowCompare" && this.state.truthId === null && this.state.attackId === null;
        const truth = autoPick ? this.truthCandidates()[0] : undefined;
        const attack = autoPick ? this.attackCandidates()[0] : undefined;
        const firstPair = mode === "mapCompare" && this.state.pairKey === null ? this.pairedCandidates()[0] : undefined;
        const firstMatch = mode === "benignVsAttack" && this.state.matchKey === null ? this.matchCandidates()[0] : undefined;
        this.update({
          mode,
          senderKey: mode === "sender" ? (this.state.senderKey ?? firstGroup) : this.state.senderKey,
          ...(truth ? { truthId: truth.id, truthWindow: 0 } : {}),
          ...(attack ? { attackId: attack.id, attackWindow: 0 } : {}),
          ...(firstPair ? { pairKey: firstPair.key } : {}),
          ...(firstMatch ? { matchKey: firstMatch.key } : {}),
        });
      }));
    this.root.querySelectorAll<HTMLButtonElement>("[data-scenario]").forEach((b) =>
      b.addEventListener("click", () => {
        const key = b.dataset.scenario || null;
        const scenario = key ? this.dataset.scenario(key) : undefined;
        const inScope = (id: number) => !scenario || scenario.benignIds.includes(id) || scenario.attackerIds.includes(id);
        const primaryId = inScope(this.state.primaryId) ? this.state.primaryId
          : (scenario ? [...scenario.benignIds, ...scenario.attackerIds][0] ?? this.state.primaryId : this.state.primaryId);
        this.update({
          scenarioKey: key, primaryId, compareId: null, senderKey: null,
          pairKey: null, pairPrimary: "real", matchKey: null, extraAttackIds: [],
          truthId: null, truthWindow: 0, attackId: null, attackWindow: 0,
        });
      }));
    this.root.querySelectorAll<HTMLSelectElement>("[data-filter]").forEach((sel) =>
      sel.addEventListener("change", () => {
        if (sel.dataset.filter === "label") this.labelFilter = sel.value;
        if (sel.dataset.filter === "group") this.groupFilter = sel.value;
        if (sel.dataset.filter === "tag") this.tagFilter = sel.value;
        this.render();
      }));
    this.root.querySelectorAll<HTMLButtonElement>('[data-role="primary"] [data-id]').forEach((b) =>
      b.addEventListener("click", () => this.update({ primaryId: Number(b.dataset.id) })));
    this.root.querySelectorAll<HTMLButtonElement>('[data-role="compare"] [data-id]').forEach((b) =>
      b.addEventListener("click", () => this.update({ compareId: Number(b.dataset.id) })));
    this.root.querySelectorAll<HTMLButtonElement>('[data-role="truth"] [data-id]').forEach((b) =>
      b.addEventListener("click", () => this.update({ truthId: Number(b.dataset.id), truthWindow: 0 })));
    this.root.querySelectorAll<HTMLButtonElement>('[data-role="attack"] [data-id]').forEach((b) =>
      b.addEventListener("click", () => this.update({ attackId: Number(b.dataset.id), attackWindow: 0 })));
    this.root.querySelectorAll<HTMLButtonElement>("[data-truthwin]").forEach((b) =>
      b.addEventListener("click", () => this.update({ truthWindow: Number(b.dataset.truthwin) })));
    this.root.querySelectorAll<HTMLButtonElement>("[data-attackwin]").forEach((b) =>
      b.addEventListener("click", () => this.update({ attackWindow: Number(b.dataset.attackwin) })));
    this.root.querySelectorAll<HTMLButtonElement>("[data-sender]").forEach((b) =>
      b.addEventListener("click", () => this.update({ senderKey: b.dataset.sender ?? null })));
    this.root.querySelectorAll<HTMLButtonElement>("[data-pair]").forEach((b) =>
      b.addEventListener("click", () => this.update({ pairKey: b.dataset.pair ?? null, pairPrimary: "real" })));
    this.root.querySelectorAll<HTMLButtonElement>("[data-match]").forEach((b) =>
      b.addEventListener("click", () => this.update({ matchKey: b.dataset.match ?? null, extraAttackIds: [] })));
    this.root.querySelectorAll<HTMLButtonElement>("[data-matchfamily]").forEach((b) =>
      b.addEventListener("click", () => {
        const family = b.dataset.matchfamily ?? "all";
        const first = this.matchCandidates().find((m) => family === "all" || m.label === family);
        this.update({ matchFamily: family, matchKey: first?.key ?? null, extraAttackIds: [] });
      }));
    this.root.querySelectorAll<HTMLInputElement>("[data-extraattack]").forEach((box) =>
      box.addEventListener("change", () => {
        const id = Number(box.dataset.extraattack);
        const rest = this.state.extraAttackIds.filter((x) => x !== id);
        this.update({ extraAttackIds: box.checked ? [...rest, id] : rest });
      }));
    this.root.querySelectorAll<HTMLButtonElement>("[data-matchprimary]").forEach((b) =>
      b.addEventListener("click", () => this.update({ matchPrimary: b.dataset.matchprimary as "benign" | "attack" })));
    this.root.querySelectorAll<HTMLButtonElement>("[data-pairprimary]").forEach((b) =>
      b.addEventListener("click", () => this.update({ pairPrimary: b.dataset.pairprimary as "real" | "attack" })));
    this.root.querySelectorAll<HTMLInputElement>("[data-toggle]").forEach((c) =>
      c.addEventListener("change", () => this.update({ [c.dataset.toggle as string]: c.checked } as Partial<ControlState>)));
    this.root.querySelector<HTMLButtonElement>('[data-action="export"]')?.addEventListener("click", () => this.handlers.onExport());
    this.root.querySelector<HTMLButtonElement>('[data-action="resetView"]')?.addEventListener("click", () => this.handlers.onResetView());
  }
}

const ATTACK_FAMILIES = ["GridSybil", "DataReplaySybil", "DoSRandomSybil", "DoSDisruptiveSybil"] as const;

function shortFamily(family: string): string {
  return family.replace(/Sybil$/, "");
}

/** "fake pseudonym #X" (GridSybil) or "N pseudonyms" (every pseudonym heard from the attacker). */
export function traceLabel(m: BenignAttackMatch): string {
  return identityTraceLabel(m.attack);
}

/** Same label, from an attack identity's own record. */
export function identityTraceLabel(attack: Identity): string {
  const r = attack.record;
  return r.attack_trace === "all_pseudonyms"
    ? `${r.broadcast_pseudonyms ?? "?"} pseudonyms`
    : `fake pseudonym #${r.sender_pseudo}`;
}

function modeLabel(m: ViewMode): string {
  switch (m) {
    case "sender": return "Sybil sender";
    case "mapCompare": return "Map compare";
    case "benignVsAttack": return "Benign vs attack";
    case "windowCompare": return "Window compare";
    default: return m;
  }
}

function modeHint(m: ViewMode): string {
  switch (m) {
    case "single": return "One pseudonym's full drive, tiled into stride-10 windows.";
    case "sender": return "All surviving pseudonyms of one physical vehicle, time-aligned — the Sybil act itself.";
    case "compare": return "Two identities side by side on the same clock (pick benign vs attacker).";
    case "mapCompare": return "One physical vehicle's real path (prepared data) next to its fabricated Sybil broadcast (raw VeReMi, never in training data) — same clock, on the map.";
    case "benignVsAttack": return "A benign vehicle and an attacker's fabricated broadcast from the same run, clipped to the same time window — three maps: benign, attack, both. Two different vehicles; neither is derived from the other.";
    case "windowCompare": return "One frozen ground-truth window next to one frozen attack window — no shared clock, step through each independently.";
  }
}
