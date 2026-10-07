import { Dataset } from "./core/dataset";
import type { Identity } from "./core/identity";
import { Playback } from "./core/playback";
import { Viewport } from "./core/viewport";
import { HeatmapRenderer } from "./render/heatmap_renderer";
import { CanvasSurface } from "./render/layer";
import { MapRenderer } from "./render/map_renderer";
import { StripRenderer } from "./render/strip_renderer";
import { ConsistencyPanel } from "./ui/consistency_panel";
import { Controls, identityTraceLabel, type ControlState } from "./ui/controls";
import { ProvenanceStrip } from "./ui/provenance";
import { SidebarResizer } from "./ui/sidebar_resizer";
import { THEME } from "./ui/theme";
import { Transport } from "./ui/transport";
import { BenignAttackWindows } from "./ui/benign_attack_windows";
import { WindowComparePanel } from "./ui/window_compare_panel";
import { WindowScrubber } from "./ui/window_scrubber";

/** Attack trail colours when several attacks share the maps; the first stays the project's attack red. */
const ATTACK_HUES = [THEME.red, THEME.orange, THEME.purple, "#EC4899", "#0EA5E9"];

/** Wires dataset → playback clock → renderers/UI. One instance per page. */
class App {
  private readonly playback = new Playback();
  private readonly map: MapRenderer;
  private readonly strips = new StripRenderer();
  private readonly heatmap: HeatmapRenderer;
  private readonly mapSurface: CanvasSurface;
  private readonly stripsSurface: CanvasSurface;
  private readonly heatmapSurface: CanvasSurface;
  private readonly consistency: ConsistencyPanel;
  private readonly provenance: ProvenanceStrip;
  private readonly controls: Controls;
  private readonly transport: Transport;
  private readonly windowScrubber: WindowScrubber;
  private readonly windowCompare: WindowComparePanel;
  /** Benign / attack / both maps for the "Benign vs attack" mode, sharing one clock and one extent. */
  private readonly triMaps: MapRenderer[];
  private readonly triSurfaces: CanvasSurface[];
  private readonly bvaWindows: BenignAttackWindows;
  private primary: Identity | null = null;
  private activeScenarioKey: string | null = null;
  private dragLast: { x: number; y: number } | null = null;

  constructor(private readonly dataset: Dataset) {
    const m = dataset.manifest;
    this.map = new MapRenderer(Viewport.fromArray(m.road_bbox));
    this.heatmap = new HeatmapRenderer({ mean: m.norm_mean, std: m.norm_std });
    // Canvases are measured after layout settles, so every resize re-renders the current frame.
    this.mapSurface = new CanvasSurface(el<HTMLCanvasElement>("map-canvas"), (w, h) => {
      this.map.resize(w, h);
      this.rerender();
    });
    this.stripsSurface = new CanvasSurface(el<HTMLCanvasElement>("strips-canvas"), () => this.rerender());
    this.heatmapSurface = new CanvasSurface(el<HTMLCanvasElement>("heatmap-canvas"), () => this.rerender());
    this.triMaps = [0, 1, 2].map(() => new MapRenderer(Viewport.fromArray(m.road_bbox)));
    this.triSurfaces = ["map-benign", "map-attack", "map-both"].map((id, i) =>
      new CanvasSurface(el<HTMLCanvasElement>(id), (w, h) => {
        this.triMaps[i].resize(w, h);
        this.rerender();
      }));
    this.triSurfaces.forEach((surface, i) => surface.setLayers([this.triMaps[i]]));
    this.mapSurface.setLayers([this.map]);
    this.stripsSurface.setLayers([this.strips]);
    this.heatmapSurface.setLayers([this.heatmap]);
    this.consistency = new ConsistencyPanel(el("consistency"));
    this.provenance = new ProvenanceStrip(el("provenance"), (label) => dataset.colorFor(label), (id) => this.controls.jumpToIdentity(id));
    this.transport = new Transport(el("transport"), this.playback, m.stride);
    this.windowScrubber = new WindowScrubber(el("window-scrubber"), (k) => this.seekToWindow(k));
    this.windowCompare = new WindowComparePanel(el("window-compare"), dataset);
    this.bvaWindows = new BenignAttackWindows(el("bva-windows"), dataset, () => this.playback.simulationTime);
    new SidebarResizer(el("sidebar-resizer"));
    this.controls = new Controls(el("sidebar"), dataset, {
      onChange: (state) => this.applyState(state),
      onExport: () => this.exportPng(),
      onResetView: () => {
        this.map.resetView();
        this.triMaps.forEach((tri) => tri.resetView());
        this.triSurfaces.forEach((surface) => surface.invalidate());
        this.mapSurface.invalidate();
        this.rerender();
      },
    });
    this.playback.onFrame((_t, step) => this.onFrame(step));
    this.bindKeyboard();
    this.bindMapNavigation();
    this.applyState(this.controls.current);
    el("status").textContent = `${dataset.identityCount} identities · ${m.total_steps.toLocaleString()} steps · seed ${m.seed}`;
  }

  private applyState(state: ControlState): void {
    const standardView = el("standard-view");
    const compareView = el("window-compare-view");
    const transportEl = el("transport");
    if (state.mode === "windowCompare") {
      standardView.hidden = true;
      compareView.hidden = false;
      transportEl.hidden = true;
      this.playback.pause();
      const sel = this.controls.windowCompareSelection();
      if (sel) {
        this.windowCompare.setSide("truth", sel.truth, sel.truthWindow);
        this.windowCompare.setSide("attack", sel.attack, sel.attackWindow);
      }
      this.windowCompare.remeasure();
      return;
    }
    standardView.hidden = false;
    compareView.hidden = true;
    transportEl.hidden = false;
    const isTriptych = state.mode === "benignVsAttack";
    const triptychEl = el("map-triptych");
    const wasHidden = triptychEl.hidden;
    triptychEl.hidden = !isTriptych;
    el("map-canvas").hidden = isTriptych;
    el("bva-windows").hidden = !isTriptych;
    standardView.classList.toggle("with-bottom", isTriptych);
    if (isTriptych && wasHidden) {
      this.triSurfaces.forEach((surface) => surface.remeasure());
      this.bvaWindows.remeasure();
    }

    const scenario = this.controls.activeScenario();
    if ((scenario?.key ?? null) !== this.activeScenarioKey) {
      this.activeScenarioKey = scenario?.key ?? null;
      this.map.setContextBox(scenario?.bbox ?? null, scenario ? `scenario extent (${scenario.subfolder})` : "benign road extent");
    }

    const scene = this.controls.scene();
    if (scene.length === 0) {
      el("map-caption").textContent = "nothing to show for this selection in the current scenario";
      return;
    }
    const primary = scene[0];
    const sceneChanged = primary !== this.primary || scene.length !== this.map.scene.length
      || scene.some((id, i) => this.map.scene[i]?.identity !== id);
    if (sceneChanged) {
      this.primary = primary;
      const inGroup = state.mode === "sender";
      const isMapCompare = state.mode === "mapCompare";
      this.map.setScene(
        scene,
        (id, i) => isMapCompare ? (id.dataSource === "raw_veremi" ? THEME.red : THEME.blue)
          : (inGroup || scene.length > 1 ? MapRenderer.pseudonymColor(i) : this.dataset.colorFor(id.label)),
        isMapCompare ? (id) => (id.dataSource === "raw_veremi" ? "attack broadcast" : "ground truth") : undefined,
      );
      this.strips.setIdentity(primary);
      this.heatmap.setIdentity(primary);
      this.consistency.setIdentity(primary);
      this.provenance.setIdentity(primary, scene.length);
      this.windowScrubber.setIdentity(primary);
      this.playback.pause();
      this.playback.setTimes(primary.times);
      if (isTriptych) this.setUpTriptych(primary);
      el("map-caption").textContent = isTriptych
        ? `${this.controls.benignAttackSelection()?.label ?? "attack"} · ${this.controls.benignAttackSelection()?.folder ?? ""} · same ${this.sharedSeconds()} s window, one clock, one map extent · green = benign vehicle · red = attacker's broadcast · two different vehicles · strips/heatmap follow the ${primary.isAttacker ? "attack" : "benign vehicle"}`
        : isMapCompare
        ? `${primary.label} · ${primary.record.family}_${primary.record.group}/${primary.record.subfolder} · vehicle ${primary.record.sender ?? "?"} · blue = ground truth (prepared data) · red = attack broadcast (raw VeReMi, never in training data) · strips/heatmap follow ${primary.dataSource === "raw_veremi" ? "the attack broadcast" : "ground truth"}`
        : scene.length > 1
          ? `${scene.length} identities · colours = pseudonyms · clock follows #${primary.record.sender_pseudo}`
          : "trail colour = |acceleration| · dashed red = teleport (>60 m/s) · dots = window starts · amber = heading · blue = velocity";
    }
    this.map.setLockToRoad(state.lockToRoad);
    this.triMaps.forEach((tri) => tri.setLockToRoad(state.lockToRoad));
    this.triSurfaces.forEach((surface) => surface.invalidate());
    this.playback.setCompressGaps(state.compressGaps);
    this.heatmap.setNormalized(state.normalized);
    this.mapSurface.invalidate();
    this.stripsSurface.invalidate();
    this.heatmapSurface.invalidate();
    this.onFrame(this.playback.stepIndex);
  }

  private sharedSeconds(): number {
    const window = this.controls.benignAttackView()?.window;
    return window ? Math.round(window[1] - window[0]) : 0;
  }

  /**
   * Map 1 = benign only, map 2 = every selected attack, map 3 = all of them; one clock, one extent
   * (the union of all trails). The bottom window-compare panel gets the same benign and attacks.
   */
  private setUpTriptych(clock: Identity): void {
    const match = this.controls.benignAttackSelection();
    const view = this.controls.benignAttackView();
    if (!match || !view) return;
    const { benign, attacks } = view;
    const attackColour = new Map(attacks.map((a, i) => [a, ATTACK_HUES[i % ATTACK_HUES.length]]));
    el("tri-title-benign").textContent = `x · Benign vehicle ${benign.record.sender ?? "?"}`;
    el("tri-title-attack").textContent = attacks.length === 1
      ? `y · ${match.label} attack — vehicle ${attacks[0].record.sender ?? "?"}, ${identityTraceLabel(attacks[0])}`
      : `y · ${attacks.length} ${match.label} attacks — vehicles ${attacks.map((a) => a.record.sender ?? "?").join(", ")}`;
    el("tri-title-both").textContent = `Both · ${match.folder}`;
    const colour = (id: Identity) => (id === benign ? THEME.green : attackColour.get(id) ?? THEME.red);
    const label = (id: Identity) => (id === benign ? "benign" : attacks.length === 1 ? "attack" : `v${id.record.sender ?? "?"}`);
    const box = [benign, ...attacks].map((id) => id.bbox()).reduce((a, b) => Viewport.union(a, b));
    const scenes = [[benign], [...attacks], [benign, ...attacks]];
    this.triMaps.forEach((tri, i) => {
      tri.setScene(scenes[i], colour, label);
      tri.setUseLayerColor(true);
      tri.setClock(clock);
      tri.setFitBox(box);
    });
    this.bvaWindows.setView(benign, attacks);
  }

  private seekToWindow(k: number): void {
    this.playback.pause();
    this.playback.seekToStep(k * this.dataset.manifest.stride);
  }

  /**
   * Mouse wheel = zoom around the cursor, drag = pan. The three benign/attack/both maps move together
   * (same extent, same zoom), so a point stays comparable across all three. Manual view resets on the
   * next scene/lock change.
   */
  private bindMapNavigation(): void {
    const groups: { canvases: HTMLCanvasElement[]; maps: MapRenderer[]; surfaces: CanvasSurface[] }[] = [
      { canvases: [el<HTMLCanvasElement>("map-canvas")], maps: [this.map], surfaces: [this.mapSurface] },
      {
        canvases: ["map-benign", "map-attack", "map-both"].map((id) => el<HTMLCanvasElement>(id)),
        maps: this.triMaps,
        surfaces: this.triSurfaces,
      },
    ];
    let dragging: (typeof groups)[number] | null = null;
    const refresh = (group: (typeof groups)[number]) => {
      group.surfaces.forEach((surface) => surface.invalidate());
      this.rerender();
    };
    for (const group of groups) {
      for (const canvas of group.canvases) {
        canvas.addEventListener("wheel", (e) => {
          e.preventDefault();
          const rect = canvas.getBoundingClientRect();
          const point = { x: e.clientX - rect.left, y: e.clientY - rect.top };
          group.maps.forEach((map) => map.zoomBy(e.deltaY < 0 ? 1.15 : 1 / 1.15, point));
          refresh(group);
        }, { passive: false });
        canvas.addEventListener("mousedown", (e) => {
          dragging = group;
          this.dragLast = { x: e.clientX, y: e.clientY };
        });
      }
    }
    window.addEventListener("mousemove", (e) => {
      if (!this.dragLast || !dragging) return;
      const dx = e.clientX - this.dragLast.x;
      const dy = e.clientY - this.dragLast.y;
      dragging.maps.forEach((map) => map.panBy(dx, dy));
      this.dragLast = { x: e.clientX, y: e.clientY };
      refresh(dragging);
    });
    window.addEventListener("mouseup", () => {
      this.dragLast = null;
      dragging = null;
    });
  }

  private rerender(): void {
    if (this.primary) this.onFrame(this.playback.stepIndex);
  }

  private onFrame(step: number): void {
    if (!this.primary) return;
    if (this.heatmap.setStep(step)) this.heatmapSurface.invalidate();
    this.mapSurface.render(step);
    if (this.controls.current.mode === "benignVsAttack") this.triSurfaces.forEach((surface) => surface.render(step));
    this.stripsSurface.render(step);
    this.heatmapSurface.render(step);
    this.consistency.update(step);
    this.provenance.update(step, this.heatmap.currentWindow);
    this.windowScrubber.setCurrentWindow(this.heatmap.currentWindow);
    this.transport.update(step);
    el("heatmap-caption").textContent = `window ${this.heatmap.currentWindow} of ${this.primary.windowCount - 1} · X_windows row ${this.primary.record.x_windows_rows[this.heatmap.currentWindow]}`;
  }

  private bindKeyboard(): void {
    window.addEventListener("keydown", (e) => {
      const target = e.target as HTMLElement | null;
      if (target && ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName)) return;
      if (document.body.dataset.tab === "encoder") return;
      const stride = this.dataset.manifest.stride;
      switch (e.key) {
        case " ": this.playback.toggle(); break;
        case "ArrowRight": this.playback.stepBy(1); break;
        case "ArrowLeft": this.playback.stepBy(-1); break;
        case "]": this.playback.stepBy(stride); break;
        case "[": this.playback.stepBy(-stride); break;
        case "Home": this.playback.seekToStep(0); break;
        case "End": this.playback.seekToStep(this.playback.stepCount - 1); break;
        case "l": case "L": this.controls.toggle("lockToRoad"); break;
        case "g": case "G": this.controls.toggle("compressGaps"); break;
        case "n": case "N": this.controls.toggle("normalized"); break;
        default: return;
      }
      e.preventDefault();
    });
  }

  private async exportPng(): Promise<void> {
    const blob = await this.mapSurface.toBlob();
    if (!blob || !this.primary) return;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `roadfm_${this.primary.record.sender_uid}_step${this.playback.stepIndex}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }
}

function el<T extends HTMLElement = HTMLElement>(id: string): T {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing #${id}`);
  return node as T;
}

/**
 * Top-level tabs: the legacy window replay and the encoder-input view. The encoder view is loaded with a
 * dynamic import on first open; `#encoder…` hashes open it directly (and are owned by the encoder view).
 */
class TabController {
  private encoderMounted: Promise<unknown> | null = null;
  private readonly legacySub: string;
  /** Last encoder view, restored when the tab is reopened. */
  private lastEncoderHash = "#encoder";

  constructor() {
    this.legacySub = document.querySelector(".topbar .sub")?.textContent ?? "";
    el("tabbar").addEventListener("click", (e) => {
      const tab = (e.target as HTMLElement).closest<HTMLElement>("[data-tab]")?.dataset.tab;
      if (tab === "encoder") {
        if (!location.hash.startsWith("#encoder")) location.hash = this.lastEncoderHash;
        else this.show("encoder");
      } else if (tab === "legacy") {
        if (location.hash.startsWith("#encoder")) {
          this.lastEncoderHash = location.hash;
          history.pushState(null, "", location.pathname + location.search);
        }
        this.show("legacy");
      }
    });
    window.addEventListener("hashchange", () => this.fromHash());
    window.addEventListener("popstate", () => this.fromHash());
    this.fromHash();
  }

  private fromHash(): void {
    this.show(location.hash.startsWith("#encoder") ? "encoder" : "legacy");
  }

  private show(tab: "legacy" | "encoder"): void {
    const isEncoder = tab === "encoder";
    document.body.dataset.tab = tab;
    document.querySelectorAll<HTMLElement>("#tabbar [data-tab]").forEach((b) => {
      b.classList.toggle("on", b.dataset.tab === tab);
      b.setAttribute("aria-selected", String(b.dataset.tab === tab));
    });
    document.querySelector(".workspace")?.toggleAttribute("hidden", isEncoder);
    el("transport").hidden = isEncoder;
    el("status").hidden = isEncoder;
    el("encoder-view").hidden = !isEncoder;
    const sub = document.querySelector(".topbar .sub");
    if (sub) {
      sub.textContent = isEncoder
        ? "What the encoder reads: 64 × 13 windows + mask, benign + GridSybil, receiver view. Sample of the full encoder input."
        : this.legacySub;
    }
    if (isEncoder && !this.encoderMounted) {
      this.encoderMounted = import("./encoder/encoder_app")
        .then(({ EncoderApp }) => EncoderApp.mount(el("encoder-view")))
        .catch((err: unknown) => {
          el("encoder-view").textContent = `failed to load the encoder view: ${err instanceof Error ? err.message : String(err)}`;
          console.error(err);
        });
    }
    // Legacy canvases were measured while hidden; nudge their ResizeObservers.
    if (!isEncoder) window.dispatchEvent(new Event("resize"));
  }
}

new TabController();

Dataset.load("data/")
  .then((ds) => new App(ds))
  .catch((err: unknown) => {
    const status = el("status");
    status.textContent = `failed to load sample: ${err instanceof Error ? err.message : String(err)}`;
    status.classList.add("error");
    console.error(err);
  });
