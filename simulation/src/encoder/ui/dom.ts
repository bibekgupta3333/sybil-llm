/** Tiny DOM helpers for the encoder view (no runtime dependencies). */

type Child = Node | string | number | null | undefined | false;

/** Create an element with attributes / properties and children. `class` and `data-*` go through attributes. */
export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number | boolean | null | undefined> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (value === true) node.setAttribute(key, "");
    else node.setAttribute(key, String(value));
  }
  append(node, ...children);
  return node;
}

export function append(parent: Node, ...children: Child[]): void {
  for (const c of children) {
    if (c === null || c === undefined || c === false) continue;
    parent.appendChild(typeof c === "string" || typeof c === "number" ? document.createTextNode(String(c)) : c);
  }
}

export function clear(node: Node): void {
  while (node.firstChild) node.removeChild(node.firstChild);
}

/** Thousands separators, fixed locale so the presentation does not depend on the laptop's settings. */
export function fmtInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}

export function fmtPct(share: number, digits = 1): string {
  return `${(share * 100).toFixed(digits)}%`;
}

/** Short form of a window id for lists: keep the tail, which carries the link + crop index. */
export function shortId(id: string, max = 28): string {
  return id.length <= max ? id : `…${id.slice(id.length - max + 1)}`;
}
