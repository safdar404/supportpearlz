"""SupportPearlz visual identity — "Prism AI": deep-space base, vibrant rainbow accents, glass cards.

Palette  space #0B0820 · glass #18123A · line #2E2560 · text #F4F1FF · muted #B9B2E0
Rainbow  #FF3D7F → #FF8A00 → #FFD600 → #22E39B → #00C2FF → #9B5CFF
Type     Space Grotesk (display) + Inter (body)
Confidence colours carry meaning: high=mint, partial=amber, none=rose.
"""
import base64
import html
from pathlib import Path

E = html.escape
ASSETS = Path(__file__).resolve().parents[2] / "assets"
LOGO_PNG = ASSETS / "agent_logo.png"
_LOGO_SVG = (ASSETS / "agent_logo.svg").read_text(encoding="utf-8")
LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(_LOGO_SVG.encode()).decode()
RAINBOW = "linear-gradient(90deg,#FF3D7F,#FF8A00,#FFD600,#22E39B,#00C2FF,#9B5CFF,#FF3D7F)"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
:root{--space:#0B0820;--glass:#18123Acc;--glass2:#211947;--line:#2E2560;--text:#F4F1FF;--muted:#B9B2E0;
--mint:#22E39B;--amber:#FFC93D;--rose:#FF5C8A;--cyan:#00C2FF;--violet:#9B5CFF;
--rainbow:""" + RAINBOW + """;}
html,body,[class*="css"],.stMarkdown,button,input,textarea,select{font-family:'Inter',system-ui,sans-serif}
h1,h2,h3,h4{font-family:'Space Grotesk','Inter',sans-serif!important;color:var(--text)!important;letter-spacing:-.01em}
h2,h3{font-size:1.15rem!important;margin:.6rem 0 .3rem!important}
.stApp{color:var(--text);background:
 radial-gradient(700px 380px at 95% -5%,#9B5CFF40,transparent 60%),
 radial-gradient(650px 360px at -5% 8%,#00C2FF30,transparent 60%),
 radial-gradient(600px 340px at 50% 110%,#FF3D7F28,transparent 60%),var(--space)!important;background-attachment:fixed!important}
.stApp::before{content:"";position:fixed;top:0;left:0;right:0;height:3px;background:var(--rainbow);background-size:200% 100%;
 animation:flow 8s linear infinite;z-index:999991}
[data-testid="stHeader"]{background:#0B0820b3!important;backdrop-filter:blur(10px)}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#120D2E,#0B0820)!important;border-right:1px solid var(--line)!important}
section[data-testid="stSidebar"] .block-container,[data-testid="stSidebarUserContent"]{padding-top:.6rem}
.block-container{padding-top:3.4rem!important;padding-bottom:3rem!important;max-width:1180px}
.stApp label,.stApp summary,[data-testid="stCaptionContainer"] p,.pz-muted{color:var(--muted)!important}
p,li{color:var(--text)}

/* hero */
.pz-hero{position:relative;display:grid;grid-template-columns:auto 1fr;gap:18px;align-items:center;padding:16px 20px;border-radius:20px;
 background:linear-gradient(135deg,#1C1545ee,#130E33ee);margin-bottom:10px;isolation:isolate}
.pz-hero::before{content:"";position:absolute;inset:0;padding:1.5px;border-radius:20px;background:var(--rainbow);background-size:200% 100%;
 animation:flow 6s linear infinite;-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
 -webkit-mask-composite:xor;mask-composite:exclude;z-index:-1}
.pz-hero img{width:84px;height:84px;filter:drop-shadow(0 6px 18px #9B5CFF66);animation:bob 4s ease-in-out infinite}
.pz-hero h1{font-size:2rem!important;line-height:1.05;margin:0 0 .2rem!important;font-weight:700;padding:0!important;
 background:var(--rainbow);background-size:200% 100%;-webkit-background-clip:text;background-clip:text;color:transparent!important;
 -webkit-text-fill-color:transparent;animation:flow 10s linear infinite}
.pz-hero p{color:var(--muted)!important;font-size:.92rem;max-width:70ch;margin:0}
.pz-tag{display:inline-block;font-size:.68rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase;padding:2px 9px;border-radius:999px;
 color:#0B0820;background:var(--rainbow);margin-bottom:6px}
.pz-chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
.pz-chip{font-size:.74rem;padding:3px 10px;border-radius:999px;background:#ffffff0d;border:1px solid var(--line);color:var(--text)}
.pz-chip.on{border-color:#22E39B88;color:#B8FFE3}
.pz-chip.demo{border-color:#FFC93D99;color:#FFE7A3}
@media(max-width:700px){.pz-hero{grid-template-columns:1fr}.pz-hero img{width:56px;height:56px}.pz-hero h1{font-size:1.6rem!important}}

/* sidebar brand */
.pz-brand{display:flex;gap:10px;align-items:center;margin:0 0 10px;padding:10px;border-radius:16px;background:#ffffff08;border:1px solid var(--line)}
.pz-brand img{width:44px;height:44px}
.pz-brand b{font-family:'Space Grotesk',Inter,sans-serif;font-size:1.1rem;background:var(--rainbow);-webkit-background-clip:text;background-clip:text;color:transparent;
 -webkit-text-fill-color:transparent;background-size:200% 100%}
.pz-brand small{display:block;color:var(--muted);font-size:.72rem}
.pz-stats{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:6px 0 4px}
.pz-stats div{background:#ffffff08;border:1px solid var(--line);border-radius:12px;padding:6px 8px;font-size:.7rem;color:var(--muted)}
.pz-stats b{display:block;font-size:1rem;color:var(--text);font-family:'Space Grotesk',Inter,sans-serif}

/* cards & widgets */
[data-testid="stExpander"],[data-testid="stMetric"],[data-testid="stForm"],.pz-answer{
 background:var(--glass)!important;border:1px solid var(--line)!important;border-radius:16px!important;backdrop-filter:blur(8px)}
[data-testid="stExpander"] details{border:0!important}
[data-testid="stMetric"]{padding:8px 12px!important}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',Inter,sans-serif;font-size:1.4rem!important}
[data-testid="stChatMessage"]{background:transparent!important;border:0!important;box-shadow:none!important;padding:4px 0!important;margin-bottom:2px;gap:.6rem}
@media(max-width:700px){[data-testid="stChatMessageAvatarCustom"],[data-testid="stChatMessageAvatarUser"]{display:none}}
.pz-answer{padding:12px 14px;position:relative;overflow:hidden}
.pz-answer::before{content:"";position:absolute;left:0;top:0;bottom:0;width:4px}
.pz-answer.high::before{background:linear-gradient(#22E39B,#00C2FF)}
.pz-answer.partial::before{background:linear-gradient(#FFD600,#FF8A00)}
.pz-answer.none::before{background:linear-gradient(#FF5C8A,#9B5CFF)}
.pill{display:inline-block;font-size:.72rem;font-weight:700;padding:2px 9px;border-radius:999px;margin:0 5px 3px 0}
.p-high{background:#22E39B22;color:#9CFFD6}.p-partial{background:#FFC93D22;color:#FFE08A}
.p-none{background:#FF5C8A22;color:#FFB3C8}.p-flag{background:#ffffff10;color:var(--muted)}.p-demo{background:#FFC93D22;color:#FFE7A3}
.cite{display:block;margin:6px 0 0;padding:7px 10px;border-radius:12px;background:#ffffff08;border:1px solid var(--line);font-size:.82rem;color:var(--text)}
.cite b{color:#9FE8FF}.cite small{color:var(--muted)}

/* buttons */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{border-radius:12px!important;font-weight:600!important;
 background:#ffffff0a!important;color:var(--text)!important;border:1px solid var(--line)!important;transition:all .15s ease}
@media(hover:hover){.stButton>button:not(:disabled):hover,.stDownloadButton>button:not(:disabled):hover{border-color:#9B5CFF!important;
 box-shadow:0 0 0 1px #9B5CFF55,0 8px 22px -10px #9B5CFFaa!important;transform:translateY(-1px)}}
button[kind="primary"],button[kind="primaryFormSubmit"]{background:linear-gradient(90deg,#FF3D7F,#FF8A00 30%,#9B5CFF)!important;
 color:#fff!important;border:0!important;box-shadow:0 8px 22px -12px #FF3D7Faa!important}
button[kind="primary"] p{color:#fff!important}
button:disabled{opacity:.45!important}
[data-testid="stBaseButton-pills"],[data-testid="stBaseButton-pillsActive"]{font-size:.8rem!important;border-color:var(--line)!important;
 background:#ffffff08!important;color:var(--text)!important}
[data-testid="stBaseButton-pills"]:hover{border-color:#00C2FF!important;color:#BFF1FF!important}

/* inputs */
[data-testid="stTextInput"] input,[data-testid="stTextArea"] textarea,[data-testid="stNumberInput"] input{background:#120D2E!important;color:var(--text)!important}
[data-baseweb="select"]>div,[data-baseweb="input"],[data-baseweb="textarea"]{background:#120D2E!important;border-color:var(--line)!important}
input::placeholder,textarea::placeholder{color:#8C84B8!important}
[data-testid="stChatInput"]{background:linear-gradient(#120D2E,#120D2E) padding-box,""" + RAINBOW + """ border-box!important;
 border:1.5px solid transparent!important;border-radius:16px!important;box-shadow:0 10px 30px -18px #00C2FF!important}
[data-testid="stChatInput"] textarea{color:var(--text)!important}
[data-testid="stBottom"]>div,[data-testid="stBottomBlockContainer"]{background:#0B0820f0!important}
[data-testid="stFileUploaderDropzone"]{background:#120D2E!important;border:1px dashed #9B5CFF88!important}

/* tabs */
.stTabs [role="tablist"]{gap:4px;background:#ffffff06;border:1px solid var(--line);border-radius:14px;padding:4px;overflow-x:auto;scrollbar-width:none}
.stTabs [role="tab"]{background:transparent!important;border-radius:10px!important;padding:6px 12px!important;color:var(--muted)!important;border:0!important;height:auto}
.stTabs [role="tab"] p{font-size:.86rem!important;font-weight:600}
.stTabs [role="tab"][aria-selected="true"]{background:linear-gradient(90deg,#FF3D7F33,#9B5CFF44,#00C2FF33)!important;color:#fff!important;box-shadow:inset 0 0 0 1px #9B5CFF77}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}

/* alerts, banners, misc */
[data-testid="stAlert"]{background:#ffffff08!important;border:1px solid var(--line)!important;border-radius:14px!important}
[data-testid="stAlert"] p{color:var(--text)!important}
[data-testid="stMarkdownContainer"] code{background:#9B5CFF22!important;color:#E2D4FF!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}
.pz-banner{padding:10px 14px;border-radius:14px;font-weight:600;margin:6px 0 10px;font-family:'Space Grotesk',Inter,sans-serif}
.b-ok{background:#22E39B18;color:#9CFFD6;border:1px solid #22E39B55}
.b-warn{background:#FFC93D18;color:#FFE08A;border:1px solid #FFC93D55}
.b-bad{background:#FF5C8A18;color:#FFB3C8;border:1px solid #FF5C8A55}
.pz-foot{text-align:center;margin-top:1.6rem;font-size:.78rem;color:var(--muted)}
.pz-foot b{background:var(--rainbow);-webkit-background-clip:text;background-clip:text;color:transparent;-webkit-text-fill-color:transparent}
.pz-doc{max-height:360px;overflow:auto;white-space:pre-wrap;font-size:.82rem;background:#120D2E;border:1px solid var(--line);border-radius:12px;padding:10px 12px;color:var(--text)}
:focus-visible{outline:2px solid #00C2FF!important;outline-offset:2px!important}
@keyframes flow{0%{background-position:0% 50%}100%{background-position:200% 50%}}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-5px)}}
@media (prefers-reduced-motion:reduce){*,*::before{animation:none!important;transition:none!important}}
</style>
"""

LOGO = f"""<div class="pz-brand"><img src="{LOGO_URI}" alt="SupportPearlz AI agent">
<div><b>SupportPearlz</b><small>🤖 AI support agent · Pearlz Home</small></div></div>"""


def hero(chips: str, demo: bool) -> str:
    tag = "🎭 Demo mode · offline" if demo else "✨ AI support agent"
    return (f'<div class="pz-hero"><img src="{LOGO_URI}" alt=""><div><span class="pz-tag">{tag}</span><h1>SupportPearlz</h1>'
            "<p>💧 Ask about AquaPearl purifiers, 🧪 filters, 🚚 delivery, 🛡️ warranty or 💸 refunds. Every answer comes from "
            "Pearlz's own documents, 📎 cites its source, and says plainly when the docs don't cover something.</p>"
            f'<div class="pz-chips">{chips}</div></div></div>')


def conf_pill(c: str) -> str:
    label = {"high": "✅ High confidence", "partial": "🟡 Partial answer", "none": "🚫 Not in documents"}.get(c, c)
    return f'<span class="pill p-{E(c)}">{E(label)}</span>'


FLAG_LABEL = {"out_of_scope": "🧭 out of scope", "injection_attempt": "🛡️ injection blocked",
              "injection_pattern_in_question": "🛡️ injection pattern", "input_truncated": "✂️ input truncated",
              "condense_failed": "⚠️ rewrite failed", "structured_output_fallback": "⚠️ fallback",
              "ungrounded_answer_suppressed": "🧯 ungrounded suppressed", "system_prompt_leak_blocked": "🔒 leak blocked"}


def answer_card(resp, demo: bool = False) -> str:
    flags = "".join(f'<span class="pill p-flag">{E(FLAG_LABEL.get(f, "⚑ " + f))}</span>'
                    for f in resp.flags if not f.startswith("pipeline"))
    if demo:
        flags += '<span class="pill p-demo">🎭 demo · extractive</span>'
    body = E(resp.answer).replace("\n", "<br>")
    unc = ""
    if resp.uncovered_parts:
        unc = '<div class="pz-muted" style="margin-top:6px;font-size:.85rem">🕳️ Not covered by the documentation: ' + E("; ".join(resp.uncovered_parts)) + "</div>"
    cites = "".join(
        f'<span class="cite">📄 <b>[{i}] {E(c.source)}</b> — {E(c.location)} <small>({E(c.doc_type)}, v{E(c.version)})</small>'
        f'<br><small>“{E(c.snippet[:180])}…”</small></span>' for i, c in enumerate(resp.sources, 1))
    if not resp.sources:
        cites = '<span class="cite">🔎 <small>No supporting source was found in the documentation.</small></span>'
    return (f'<div class="pz-answer {E(resp.confidence.value)}">{conf_pill(resp.confidence.value)}{flags}'
            f'<div style="margin-top:6px;line-height:1.55">{body}</div>{unc}<div style="margin-top:6px">{cites}</div></div>')


def transcript_md(chat, session_id: str) -> str:
    """Markdown export of the conversation with citations."""
    out = [f"# 🤖 SupportPearlz conversation · session {session_id}\n"]
    for i, (q, r) in enumerate(chat, 1):
        out.append(f"## {i}. 🙂 {q}\n\n**🤖 Answer** ({r.confidence.value} confidence)\n\n{r.answer}\n")
        for j, c in enumerate(r.sources, 1):
            out.append(f"- 📄 [{j}] {c.source} — {c.location} (v{c.version})")
        out.append("")
    return "\n".join(out)
