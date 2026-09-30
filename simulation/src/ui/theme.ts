/** Design tokens — Power BI-style analytics palette, kept in sync with tracker.html. */
export const THEME = {
  bg: "#F8FAFC",
  panel: "#FFFFFF",
  card: "#F1F5F9", // label-chip background — distinct from the white panel so chips stay legible
  border: "#E2E8F0",
  borderSoft: "#E2E8F0",
  text: "#475569",
  textStrong: "#0F172A",
  muted: "#64748B",
  dim: "#94A3B8",
  blue: "#2563EB",
  blueDeep: "#1E3A8A",
  purple: "#8B5CF6",
  orange: "#F97316",
  green: "#16A34A",
  red: "#DC2626",
  amber: "#F59E0B",
  gridLine: "rgba(100,116,139,0.18)",
  windowBand: "rgba(37,99,235,0.08)",
  windowBandAlt: "rgba(139,92,246,0.08)",
  roadBox: "rgba(22,163,74,0.45)",
} as const;

/** Sequential colour for acceleration magnitude, 0 → calm, 1 → hard braking/acceleration. */
export function accelColor(t: number): string {
  const x = Math.min(1, Math.max(0, t));
  const stops: readonly [number, number, number][] = [
    [37, 99, 235], // primary blue, calm
    [245, 158, 11], // amber, moderate
    [220, 38, 38], // red, hard braking/acceleration
  ];
  const seg = x < 0.5 ? 0 : 1;
  const localT = x < 0.5 ? x * 2 : (x - 0.5) * 2;
  const [r0, g0, b0] = stops[seg];
  const [r1, g1, b1] = stops[seg + 1];
  const r = Math.round(r0 + (r1 - r0) * localT);
  const g = Math.round(g0 + (g1 - g0) * localT);
  const b = Math.round(b0 + (b1 - b0) * localT);
  return `rgb(${r},${g},${b})`;
}

export function withAlpha(hex: string, alpha: number): string {
  const n = parseInt(hex.replace("#", ""), 16);
  const r = (n >> 16) & 255;
  const g = (n >> 8) & 255;
  const b = n & 255;
  return `rgba(${r},${g},${b},${alpha})`;
}
