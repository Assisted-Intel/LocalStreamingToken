#!/usr/bin/env python3
"""
Local Streaming Token — mobile layout check.

Loads the running app in a phone-sized browser and asserts the things that are easy to
break and impossible to notice from a desktop:

  * no tab page scrolls sideways,
  * every section is reachable through the drawer,
  * nothing interactive is smaller than a fingertip,
  * no text field is under 16px, which is what makes iOS zoom on focus,
  * no JavaScript errors.

Not part of ``pytest``: it needs a running server and a real browser, and a browser test
in the unit suite would be slow and flaky. Run it by hand after touching
``static/mobile.css``, ``static/index.html`` or the layout parts of ``static/app.js``.

    python main.py                      # in one terminal
    python tools/mobile_check.py http://127.0.0.1:8756 --password admin

Playwright is already a dependency (see requirements.txt); if the browser itself is
missing, run ``playwright install chromium`` once.
"""

import argparse
import sys

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("playwright is not installed — pip install playwright && playwright install chromium")

TABS = ["chat", "batch", "evals", "database", "resources", "personas", "memory", "settings"]

# One pass over the live DOM. Everything is measured from the rendered box, not from the
# stylesheet, so a rule that is present but overridden still counts as broken.
MEASURE = """
() => {
  const de = document.documentElement, vw = de.clientWidth;
  const name = (el) => el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
    (typeof el.className === 'string' && el.className
       ? '.' + el.className.trim().split(/\\s+/)[0] : '');
  // A wide element inside a deliberate horizontal scroller (the composer strip, the data
  // grids) is fine; one that widens the page is not.
  const inScroller = (el) => {
    for (let p = el.parentElement; p; p = p.parentElement) {
      const o = getComputedStyle(p).overflowX;
      if (o === 'auto' || o === 'scroll') return true;
    }
    return false;
  };
  const wide = [];
  document.querySelectorAll('*').forEach((el) => {
    const r = el.getBoundingClientRect();
    if ((!r.width && !r.height) || getComputedStyle(el).display === 'none') return;
    if (r.right <= vw + 1) return;
    const pr = el.parentElement && el.parentElement.getBoundingClientRect();
    if (pr && pr.right > vw + 1) return;      // report the outermost cause only
    if (inScroller(el)) return;
    wide.push({ sel: name(el), w: Math.round(r.width), right: Math.round(r.right) });
  });
  const small = [], zoom = [];
  document.querySelectorAll('button, .tab, summary, a[href]').forEach((el) => {
    const r = el.getBoundingClientRect();
    if (r.width && r.height && (r.height < 40 || r.width < 28)) {
      small.push({ sel: name(el), w: Math.round(r.width), h: Math.round(r.height) });
    }
  });
  document.querySelectorAll('input, select, textarea').forEach((el) => {
    const r = el.getBoundingClientRect();
    if (!r.width && !r.height) return;
    if (el.type === 'checkbox' || el.type === 'radio' || el.type === 'range') return;
    const fs = parseFloat(getComputedStyle(el).fontSize);
    if (fs < 16) zoom.push({ sel: name(el), fs });
  });
  return { overflow: de.scrollWidth - vw, wide, small, zoom };
}
"""


def main():
    ap = argparse.ArgumentParser(description="Check the app at phone size.")
    ap.add_argument("base", nargs="?", default="http://127.0.0.1:8756",
                    help="where the app is running (default: %(default)s)")
    ap.add_argument("--username", default="admin")
    ap.add_argument("--password", default="admin")
    ap.add_argument("--width", type=int, default=390)
    ap.add_argument("--height", type=int, default=844)
    ap.add_argument("--shots", help="directory to write a screenshot of each tab into")
    args = ap.parse_args()

    failures = []

    def check(label, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
        if not ok:
            failures.append(label)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        ctx = browser.new_context(viewport={"width": args.width, "height": args.height},
                                  device_scale_factor=2, is_mobile=True, has_touch=True)
        page = ctx.new_page()
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))

        page.goto(f"{args.base}/login", wait_until="networkidle")
        page.fill("#username", args.username)
        page.fill("#password", args.password)
        page.click("#login-btn")
        page.wait_for_timeout(1500)
        if page.is_visible("#skip-btn"):
            page.click("#skip-btn")        # the change-your-password nag
        page.wait_for_url(lambda u: "/login" not in u, timeout=20000)
        page.wait_for_timeout(1500)

        print("\nNavigation")
        check("the tab strip moved into the drawer",
              page.evaluate("document.querySelector('nav.tabs').parentElement.id") == "nav-drawer")
        page.tap("#btn-nav")
        page.wait_for_timeout(400)
        for t in TABS:
            r = page.evaluate("""(t) => {
                const el = document.querySelector(`[data-tab=${t}]`);
                const b = el.getBoundingClientRect();
                const hit = document.elementFromPoint(b.left + b.width / 2, b.top + b.height / 2);
                return { on: b.left >= 0 && b.right <= innerWidth && b.top >= 0,
                         reach: !!hit && (el === hit || el.contains(hit)) };
            }""", t)
            check(f"{t} is reachable", r["on"] and r["reach"])
        page.tap("[data-tab=chat]")
        page.wait_for_timeout(500)

        for tab in TABS:
            page.evaluate(f"document.querySelector('[data-tab={tab}]').click()")
            page.wait_for_timeout(700)
            m = page.evaluate(MEASURE)
            print(f"\n{tab}")
            check("does not scroll sideways", m["overflow"] == 0, f"{m['overflow']}px")
            check("nothing is laid out off-screen", not m["wide"],
                  "; ".join(f"{w['sel']} right={w['right']}" for w in m["wide"][:4]))
            check("touch targets are big enough", not m["small"],
                  "; ".join(f"{s['sel']} {s['w']}x{s['h']}" for s in m["small"][:4]))
            check("no field small enough to trigger iOS zoom", not m["zoom"],
                  "; ".join(f"{z['sel']} {z['fs']}px" for z in m["zoom"][:4]))
            if args.shots:
                page.screenshot(path=f"{args.shots}/mobile-{tab}.png")

        print()
        check("no JavaScript errors", not errors, "; ".join(errors[:3]))
        browser.close()

    print(f"\n{len(failures)} failing check(s)" if failures else "\nAll checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
