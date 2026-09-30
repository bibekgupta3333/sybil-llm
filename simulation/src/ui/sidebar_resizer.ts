const MIN_WIDTH = 260;
const MAX_WIDTH = 640;
const DEFAULT_WIDTH = 300;
const STORAGE_KEY = "roadfm.sidebarWidth";

/** Clamp a candidate sidebar width to the usable range. Pure, so it's testable without a DOM. */
export function clampSidebarWidth(px: number): number {
  return Math.min(MAX_WIDTH, Math.max(MIN_WIDTH, Math.round(px)));
}

/**
 * Drag-to-resize the sidebar: a thin handle between it and the stage. Width lives in a CSS custom
 * property (`--sidebar-w` on the document root) so every layout rule that reads it just works, and
 * persists to `localStorage` so it survives a reload. Double-click resets to the default width.
 */
export class SidebarResizer {
  private dragging = false;

  constructor(handle: HTMLElement) {
    this.setWidth(SidebarResizer.storedWidth() ?? DEFAULT_WIDTH);

    handle.addEventListener("mousedown", (e) => {
      this.dragging = true;
      handle.classList.add("active");
      e.preventDefault();
    });
    handle.addEventListener("dblclick", () => this.setWidth(DEFAULT_WIDTH));
    window.addEventListener("mousemove", (e) => {
      if (!this.dragging) return;
      const workspaceLeft = handle.parentElement?.getBoundingClientRect().left ?? 0;
      this.setWidth(e.clientX - workspaceLeft);
    });
    window.addEventListener("mouseup", () => {
      this.dragging = false;
      handle.classList.remove("active");
    });
  }

  private static storedWidth(): number | null {
    try {
      const raw = Number(localStorage.getItem(STORAGE_KEY));
      return Number.isFinite(raw) && raw > 0 ? raw : null;
    } catch {
      return null; // private browsing / storage disabled: fall back to the default
    }
  }

  private setWidth(px: number): void {
    const clamped = clampSidebarWidth(px);
    document.documentElement.style.setProperty("--sidebar-w", `${clamped}px`);
    try {
      localStorage.setItem(STORAGE_KEY, String(clamped));
    } catch {
      // private browsing / storage disabled: width just won't survive a reload
    }
  }
}
