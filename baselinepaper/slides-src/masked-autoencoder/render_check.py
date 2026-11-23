import sys, json
from pathlib import Path
from playwright.sync_api import sync_playwright
deck = Path(sys.argv[1]).resolve(); out = Path(sys.argv[2]); out.mkdir(exist_ok=True)
JS = """() => {
 const s = document.querySelector('.slide.active'); const sr = s.getBoundingClientRect(); const probs = [];
 for (const el of s.querySelectorAll('*')) {
   if (el.closest('.notes') || el.closest('svg') && el.tagName !== 'svg') continue;
   const r = el.getBoundingClientRect(); if (!r.width) continue;
   if (r.right > sr.right + 1 || r.bottom > sr.bottom + 1 || r.left < sr.left - 1) probs.push(el.tagName + '.' + el.className + ' ' + (el.textContent||'').slice(0,40));
   if (el.scrollWidth > el.clientWidth + 2 && getComputedStyle(el).overflow !== 'visible' && el !== s) probs.push('scroll ' + el.tagName);
 }
 // foot overlap with content above
 const foot = s.querySelector('.foot'); if (foot) { const fr = foot.getBoundingClientRect();
   for (const el of s.children) { if (el === foot || el.classList.contains('notes')) continue; const r = el.getBoundingClientRect();
     if (r.height && r.bottom > fr.top + 2 && r.top < fr.top) probs.push('foot overlap ' + el.tagName + '.' + el.className); } }
 // svg text overlap
 const texts = [...s.querySelectorAll('svg text')].map(t => [t, t.getBoundingClientRect()]);
 for (let i = 0; i < texts.length; i++) for (let j = i + 1; j < texts.length; j++) { const a = texts[i][1], b = texts[j][1];
   const ox = Math.min(a.right, b.right) - Math.max(a.left, b.left), oy = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
   if (ox > 2 && oy > 2) probs.push('text overlap: ' + texts[i][0].textContent + ' | ' + texts[j][0].textContent); }
 return probs; }"""
with sync_playwright() as p:
    b = p.chromium.launch(channel="chrome")
    for theme in ("light", "dark"):
        pg = b.new_page(viewport={"width": 1280, "height": 720}, color_scheme=theme)
        pg.goto(f"file://{deck}?theme={theme}")
        n = pg.evaluate("document.querySelectorAll('.slide').length")
        for k in range(1, n + 1):
            pg.goto(f"file://{deck}?theme={theme}#{k}"); pg.reload(); pg.wait_for_timeout(80)
            probs = pg.evaluate(JS)
            if probs: print(theme, k, probs[:6])
            pg.screenshot(path=str(out / f"{theme}-{k:02d}.png"))
        print(theme, "slides", n)
    b.close()
