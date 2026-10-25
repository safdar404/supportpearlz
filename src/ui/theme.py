"""SupportPearlz visual identity: mother-of-pearl light theme with deep-water teal.

Palette  pearl #F6F8FC · lilac #EDE7FF · aqua #DDF8F4 · teal #0E8C86 · deep #0B5D5A · navy #13233A · coral #F06A5D · amber #E89B2A
Type     Outfit (display) + Nunito Sans (body)
Confidence colours carry meaning: high=teal, partial=amber, none=coral.
"""
import html

E = html.escape

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@500;600;700;800&family=Nunito+Sans:wght@400;600;700&display=swap');
:root{--pearl:#F6F8FC;--lilac:#EDE7FF;--aqua:#DDF8F4;--blush:#FFEAF0;--teal:#0E8C86;--deep:#0B5D5A;--navy:#13233A;
--muted:#5B6B82;--line:#DCE3EE;--coral:#F06A5D;--amber:#E89B2A;--violet:#7B6CF6;--card:#FFFFFF;}
html,body,[class*="css"],.stMarkdown,button,input,textarea{font-family:'Nunito Sans',system-ui,sans-serif;}
h1,h2,h3,h4{font-family:'Outfit','Nunito Sans',sans-serif !important;color:var(--navy);letter-spacing:-.01em}
.stApp{background:
 radial-gradient(900px 420px at 100% -5%,rgba(123,108,246,.16),transparent 60%),
 radial-gradient(800px 420px at -5% 10%,rgba(14,140,134,.14),transparent 60%),
 radial-gradient(700px 400px at 50% 105%,rgba(240,106,93,.08),transparent 60%),var(--pearl);}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#FFFFFF,#EEF6F7);border-right:1px solid var(--line)}
.block-container{padding-top:1.4rem;max-width:1250px}
.pz-hero{display:grid;grid-template-columns:1.3fr .9fr;gap:20px;align-items:center;padding:22px 28px;border-radius:24px;
 background:linear-gradient(120deg,#FFFFFF 0%,#F3EEFF 45%,#E2FAF6 100%);border:1px solid var(--line);
 box-shadow:0 10px 30px -18px rgba(19,35,58,.35);margin-bottom:12px}
.pz-hero h1{font-size:2.6rem;line-height:1.02;margin:0 0 .35rem;font-weight:800;
 background:linear-gradient(90deg,var(--deep),var(--teal) 45%,var(--violet));-webkit-background-clip:text;background-clip:text;color:transparent}
.pz-hero p{color:var(--muted);font-size:1.02rem;max-width:58ch;margin:0}
.pz-chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.pz-chip{font-size:.8rem;padding:4px 11px;border-radius:999px;background:#fff;border:1px solid var(--line);color:var(--navy)}
@media(max-width:900px){.pz-hero{grid-template-columns:1fr}.pz-hero svg{display:none}}
.pz-answer{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px 16px;box-shadow:0 6px 18px -14px rgba(19,35,58,.4)}
.pz-answer.none{border-left:5px solid var(--coral)}
.pz-answer.partial{border-left:5px solid var(--amber)}
.pz-answer.high{border-left:5px solid var(--teal)}
.pill{display:inline-block;font-size:.74rem;font-weight:700;padding:2px 10px;border-radius:999px;margin-right:6px}
.p-high{background:rgba(14,140,134,.12);color:var(--deep)}
.p-partial{background:rgba(232,155,42,.15);color:#8A5A0E}
.p-none{background:rgba(240,106,93,.13);color:#A3352A}
.p-flag{background:#EEF1F7;color:var(--muted)}
.cite{display:block;margin:6px 0 0;padding:8px 10px;border-radius:12px;background:linear-gradient(90deg,#F2FBFA,#F6F3FF);border:1px solid var(--line);font-size:.86rem}
.cite b{color:var(--deep)}
.cite small{color:var(--muted)}
.pz-stat{background:#fff;border:1px solid var(--line);border-radius:14px;padding:10px 12px}
div[data-testid="stMetric"]{background:#fff;border:1px solid var(--line);border-radius:14px;padding:10px 14px}
.stTabs [data-baseweb="tab"]{background:#fff;border-radius:12px 12px 0 0;padding:8px 14px;border:1px solid var(--line);border-bottom:0}
.stTabs [aria-selected="true"]{background:linear-gradient(180deg,#E2FAF6,#fff)}
.stButton>button{border-radius:12px;font-weight:700}
.stButton>button[kind="primary"]{background:linear-gradient(90deg,var(--teal),var(--deep));border:0}
.stButton>button:focus-visible{outline:2px solid var(--violet);outline-offset:2px}
.pz-muted{color:var(--muted)}
.pz-banner{padding:12px 16px;border-radius:14px;font-weight:700;margin:6px 0 10px;font-family:'Outfit'}
.b-ok{background:rgba(14,140,134,.1);color:var(--deep);border:1px solid rgba(14,140,134,.35)}
.b-warn{background:rgba(232,155,42,.12);color:#8A5A0E;border:1px solid rgba(232,155,42,.4)}
.b-bad{background:rgba(240,106,93,.1);color:#A3352A;border:1px solid rgba(240,106,93,.4)}
@media (prefers-reduced-motion:reduce){*{animation:none!important}}
.drop{animation:bob 4s ease-in-out infinite}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
</style>
"""

HERO_SVG = """
<svg viewBox="0 0 360 220" width="100%" role="img" aria-label="Water droplet with a pearl and document pages">
 <defs>
  <linearGradient id="w" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5FD3C9"/><stop offset="1" stop-color="#0B5D5A"/></linearGradient>
  <radialGradient id="pearl" cx=".35" cy=".3"><stop offset="0" stop-color="#fff"/><stop offset=".5" stop-color="#EDE7FF"/><stop offset="1" stop-color="#B9B0F5"/></radialGradient>
 </defs>
 <g opacity=".9">
  <rect x="34" y="52" width="92" height="118" rx="12" fill="#fff" stroke="#DCE3EE"/>
  <rect x="48" y="70" width="58" height="6" rx="3" fill="#0E8C86" opacity=".7"/>
  <rect x="48" y="86" width="64" height="5" rx="2.5" fill="#C9D3E2"/><rect x="48" y="98" width="52" height="5" rx="2.5" fill="#C9D3E2"/>
  <rect x="48" y="110" width="60" height="5" rx="2.5" fill="#C9D3E2"/><rect x="48" y="130" width="40" height="14" rx="7" fill="#DDF8F4"/>
  <rect x="234" y="40" width="92" height="118" rx="12" fill="#fff" stroke="#DCE3EE" transform="rotate(6 280 99)"/>
  <rect x="250" y="64" width="54" height="6" rx="3" fill="#7B6CF6" opacity=".6" transform="rotate(6 280 99)"/>
  <rect x="250" y="80" width="62" height="5" rx="2.5" fill="#C9D3E2" transform="rotate(6 280 99)"/>
  <rect x="250" y="92" width="48" height="5" rx="2.5" fill="#C9D3E2" transform="rotate(6 280 99)"/>
 </g>
 <g class="drop">
  <path d="M180 22 C150 70,128 100,128 132 a52 52 0 0 0 104 0 C232 100,210 70,180 22Z" fill="url(#w)"/>
  <circle cx="180" cy="134" r="27" fill="url(#pearl)"/>
  <ellipse cx="170" cy="124" rx="7" ry="4" fill="#fff" opacity=".9"/>
 </g>
 <path d="M126 112 C140 112,140 130,152 130" stroke="#0E8C86" stroke-width="2" fill="none" stroke-dasharray="4 4"/>
 <path d="M234 104 C220 104,220 126,208 126" stroke="#7B6CF6" stroke-width="2" fill="none" stroke-dasharray="4 4"/>
 <text x="180" y="208" text-anchor="middle" font-family="Nunito Sans" font-size="11" fill="#5B6B82">answers drawn from your own documents</text>
</svg>
"""

LOGO = """<div style="display:flex;gap:10px;align-items:center;margin-bottom:8px">
<svg width="36" height="36" viewBox="0 0 36 36"><defs><linearGradient id="lw" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#5FD3C9"/>
<stop offset="1" stop-color="#0B5D5A"/></linearGradient></defs><path d="M18 3 C11 13,7 19,7 24 a11 11 0 0 0 22 0 C29 19,25 13,18 3Z" fill="url(#lw)"/>
<circle cx="18" cy="24" r="5.5" fill="#EDE7FF"/><circle cx="16.5" cy="22.5" r="1.6" fill="#fff"/></svg>
<div><div style="font-family:Outfit;font-weight:800;font-size:1.15rem;color:#13233A">SupportPearlz</div>
<div style="color:#5B6B82;font-size:.75rem">Pearlz Home Systems · support desk</div></div></div>"""


def conf_pill(c: str) -> str:
    label = {"high": "✅ High confidence", "partial": "🟡 Partial answer", "none": "🔴 Not in documents"}.get(c, c)
    return f'<span class="pill p-{E(c)}">{E(label)}</span>'


def answer_card(resp) -> str:
    flags = "".join(f'<span class="pill p-flag">⚑ {E(f)}</span>' for f in resp.flags if not f.startswith("pipeline"))
    body = E(resp.answer).replace("\n", "<br>")
    unc = ""
    if resp.uncovered_parts:
        unc = '<div class="pz-muted" style="margin-top:6px">Not covered by the documentation: ' + E("; ".join(resp.uncovered_parts)) + "</div>"
    cites = "".join(
        f'<span class="cite">📄 <b>[{i}] {E(c.source)}</b> — {E(c.location)} <small>({E(c.doc_type)}, v{E(c.version)})</small>'
        f'<br><small>“{E(c.snippet[:200])}…”</small></span>' for i, c in enumerate(resp.sources, 1))
    if not resp.sources:
        cites = '<span class="cite">🔎 <small>No supporting source was found in the documentation.</small></span>'
    return (f'<div class="pz-answer {E(resp.confidence.value)}">{conf_pill(resp.confidence.value)}{flags}'
            f'<div style="margin-top:8px;line-height:1.55">{body}</div>{unc}<div style="margin-top:8px">{cites}</div></div>')
