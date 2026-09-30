import type { Playback } from "../core/playback";

/** Footer: play/pause, stepping, speed, scrub bar, readouts. */
export class Transport {
  private readonly playBtn: HTMLButtonElement;
  private readonly slider: HTMLInputElement;
  private readonly readout: HTMLElement;

  constructor(root: HTMLElement, private readonly playback: Playback, private readonly stride: number) {
    root.innerHTML = `
      <button class="btn" data-t="start" title="Home">⇤</button>
      <button class="btn" data-t="prevWin" title="[ — previous window">⏮</button>
      <button class="btn" data-t="prev" title="← previous step">◂</button>
      <button class="btn primary" data-t="play" title="Space">▶ Play</button>
      <button class="btn" data-t="next" title="→ next step">▸</button>
      <button class="btn" data-t="nextWin" title="] — next window">⏭</button>
      <label class="speed">speed
        <select data-t="speed">${[0.25, 0.5, 1, 2, 4, 8].map((s) => `<option value="${s}" ${s === 1 ? "selected" : ""}>×${s}</option>`).join("")}</select>
      </label>
      <input type="range" min="0" max="1000" value="0" data-t="scrub" class="scrub" aria-label="scrub">
      <span class="readout" data-t="readout"></span>`;
    this.playBtn = root.querySelector('[data-t="play"]')!;
    this.slider = root.querySelector('[data-t="scrub"]')!;
    this.readout = root.querySelector('[data-t="readout"]')!;

    root.querySelector('[data-t="start"]')!.addEventListener("click", () => playback.seekToStep(0));
    root.querySelector('[data-t="prev"]')!.addEventListener("click", () => playback.stepBy(-1));
    root.querySelector('[data-t="next"]')!.addEventListener("click", () => playback.stepBy(1));
    root.querySelector('[data-t="prevWin"]')!.addEventListener("click", () => playback.stepBy(-stride));
    root.querySelector('[data-t="nextWin"]')!.addEventListener("click", () => playback.stepBy(stride));
    this.playBtn.addEventListener("click", () => playback.toggle());
    root.querySelector<HTMLSelectElement>('[data-t="speed"]')!.addEventListener("change", (e) =>
      playback.setSpeed(Number((e.target as HTMLSelectElement).value)));
    this.slider.addEventListener("input", () => {
      playback.pause();
      playback.seekToProgress(Number(this.slider.value) / 1000);
    });
  }

  update(step: number): void {
    const pb = this.playback;
    this.playBtn.textContent = pb.playing ? "⏸ Pause" : "▶ Play";
    this.playBtn.classList.toggle("playing", pb.playing);
    if (document.activeElement !== this.slider) this.slider.value = String(Math.round(pb.progress * 1000));
    const win = Math.floor(step / this.stride);
    this.readout.textContent = `step ${step} / ${Math.max(0, pb.stepCount - 1)} · window ${win} · t = ${pb.simulationTime.toFixed(2)} s`;
  }
}
