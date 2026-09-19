"""Visual theme: animated light/dark green design system."""

PALETTES = {
    "light": dict(
        bg1="#f4fff7", bg2="#e3f8ea", bg3="#d2f3df", bg4="#eefcf2",
        surface="rgba(255,255,255,.66)", border="rgba(255,255,255,.85)",
        ink="#12372a", muted="#5b7c6d",
        g100="#dcf5e3", g200="#b9ebc9", g300="#8fdcaa",
        g400="#5fc98a", g500="#34b26b", g600="#22935a", g700="#1b7449",
        sidebar1="#e9fbef", sidebar2="#d3f4df", plot="rgba(255,255,255,.55)",
    ),
    "dark": dict(
        bg1="#07130e", bg2="#0c1f17", bg3="#102a1f", bg4="#0a1a13",
        surface="rgba(18,45,34,.62)", border="rgba(95,201,138,.22)",
        ink="#e6fff0", muted="#8fb5a1",
        g100="#16362a", g200="#1e4b39", g300="#2f7a58",
        g400="#4dbd86", g500="#5fd79a", g600="#7ee6b0", g700="#a5f3c9",
        sidebar1="#0b1f17", sidebar2="#0e2a1f", plot="rgba(12,31,23,.55)",
    ),
}


def get_css(mode: str = "light") -> str:
    p = PALETTES.get(mode, PALETTES["light"])
    v = "\n".join(f"  --{k}:{val};" for k, val in p.items())
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@500;700&display=swap');

:root {{
{v}
}}

html, body, [class*="css"], .stApp {{ font-family:'Inter',sans-serif !important; color:var(--ink); }}
h1,h2,h3,h4,.kpi-value {{ font-family:'Sora',sans-serif !important; }}

.stApp {{
  background: linear-gradient(135deg,var(--bg1),var(--bg2),var(--bg3),var(--bg4));
  background-size:400% 400%; animation:bgShift 20s ease infinite;
}}
@keyframes bgShift {{0%{{background-position:0% 50%}}50%{{background-position:100% 50%}}100%{{background-position:0% 50%}}}}

.stApp::before, .stApp::after {{
  content:""; position:fixed; border-radius:50%; filter:blur(80px); opacity:.40; z-index:0; pointer-events:none;
}}
.stApp::before {{ width:420px;height:420px;background:var(--g300);top:-120px;right:-90px;animation:fl1 16s ease-in-out infinite; }}
.stApp::after  {{ width:340px;height:340px;background:var(--g200);bottom:-110px;left:-70px;animation:fl2 19s ease-in-out infinite; }}
@keyframes fl1 {{0%,100%{{transform:translate(0,0) scale(1)}}50%{{transform:translate(-60px,70px) scale(1.12)}}}}
@keyframes fl2 {{0%,100%{{transform:translate(0,0) scale(1)}}50%{{transform:translate(70px,-60px) scale(1.1)}}}}

header[data-testid="stHeader"] {{ background:transparent; }}
.block-container {{ padding-top:1.6rem; max-width:1320px; position:relative; z-index:1; animation:fadeUp .8s cubic-bezier(.2,.8,.2,1) both; }}
@keyframes fadeUp {{from{{opacity:0;transform:translateY(30px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes pop {{from{{opacity:0;transform:scale(.88) translateY(14px)}}to{{opacity:1;transform:scale(1) translateY(0)}}}}
@keyframes pulse {{0%,100%{{box-shadow:0 0 0 0 rgba(52,178,107,.45)}}70%{{box-shadow:0 0 0 18px rgba(52,178,107,0)}}}}
@keyframes sheen {{0%{{background-position:-220% 0}}100%{{background-position:220% 0}}}}

.hero {{
  background:linear-gradient(120deg,var(--g600),var(--g500),var(--g400));
  background-size:220% 220%; animation:bgShift 12s ease infinite;
  border-radius:30px; padding:2.5rem 2.4rem; color:#06231a; position:relative; overflow:hidden;
  box-shadow:0 22px 55px rgba(34,147,90,.30); margin-bottom:1.4rem;
}}
.hero h1 {{ font-size:2.7rem; font-weight:800; margin:0; letter-spacing:-1px; color:#06231a; }}
.hero p {{ font-weight:300; font-size:1.05rem; margin:.55rem 0 0; max-width:720px; opacity:.88; }}
.hero .badge {{ display:inline-block; background:rgba(255,255,255,.28); backdrop-filter:blur(6px);
  padding:.3rem .9rem; border-radius:999px; font-size:.76rem; font-weight:600; margin:0 .4rem .85rem 0; }}
.hero::after {{ content:"◆"; position:absolute; right:2.2rem; top:.6rem; font-size:7rem; opacity:.14; }}

.glass {{
  background:var(--surface); backdrop-filter:blur(16px); border:1px solid var(--border);
  border-radius:22px; padding:1.35rem 1.45rem; box-shadow:0 10px 34px rgba(0,0,0,.08);
  transition:transform .35s cubic-bezier(.2,.8,.2,1), box-shadow .35s ease; animation:pop .55s ease both;
  height:100%;
}}
.glass:hover {{ transform:translateY(-7px); box-shadow:0 22px 46px rgba(34,147,90,.22); }}

.kpi-label {{ font-size:.72rem; letter-spacing:1.4px; text-transform:uppercase; color:var(--muted); font-weight:600; }}
.kpi-value {{ font-size:2rem; font-weight:700; color:var(--g700); font-family:'JetBrains Mono',monospace !important; line-height:1.25; }}
.kpi-sub {{ font-size:.8rem; color:var(--muted); }}
.kpi-value.sheen {{
  background:linear-gradient(90deg,var(--g700),var(--g400),var(--g700));
  background-size:220% auto; -webkit-background-clip:text; background-clip:text; color:transparent;
  animation:sheen 3.2s linear infinite;
}}

.tag {{ display:inline-block; padding:.32rem 1rem; border-radius:999px; font-weight:700; font-size:.82rem;
  color:#06231a; animation:pulse 2.4s infinite; }}

.stTabs [data-baseweb="tab-list"] {{ gap:.4rem; background:var(--surface); padding:.4rem; border-radius:999px; backdrop-filter:blur(10px); }}
.stTabs [data-baseweb="tab"] {{ border-radius:999px; padding:.5rem 1.15rem; font-weight:500; transition:all .3s ease; }}
.stTabs [data-baseweb="tab"]:hover {{ background:var(--g100); transform:translateY(-2px); }}
.stTabs [aria-selected="true"] {{ background:linear-gradient(120deg,var(--g500),var(--g400)) !important; color:#06231a !important; box-shadow:0 8px 18px rgba(52,178,107,.35); }}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}

.stButton>button, .stDownloadButton>button {{
  background:linear-gradient(120deg,var(--g600),var(--g400)); color:#06231a; border:none; border-radius:15px;
  padding:.7rem 1.3rem; font-weight:600; font-family:'Inter',sans-serif;
  box-shadow:0 8px 20px rgba(52,178,107,.32); transition:all .3s cubic-bezier(.2,.8,.2,1);
}}
.stButton>button:hover, .stDownloadButton>button:hover {{ transform:translateY(-3px) scale(1.02); box-shadow:0 16px 30px rgba(52,178,107,.45); color:#06231a; }}
.stButton>button:active {{ transform:scale(.97); }}

.stNumberInput input, .stTextInput input, .stSelectbox div[data-baseweb="select"]>div {{
  border-radius:13px !important; border:1.5px solid var(--g200) !important;
  background:var(--surface) !important; color:var(--ink) !important; transition:all .25s ease;
}}
.stNumberInput input:focus, .stTextInput input:focus {{ border-color:var(--g500) !important; box-shadow:0 0 0 4px rgba(52,178,107,.2) !important; }}
div[data-testid="stSlider"] [role="slider"] {{ background:var(--g500) !important; }}

section[data-testid="stSidebar"] {{ background:linear-gradient(180deg,var(--sidebar1),var(--sidebar2)); border-right:1px solid var(--g200); }}
section[data-testid="stSidebar"] * {{ color:var(--ink); }}

div[data-testid="stDataFrame"] {{ border-radius:18px; overflow:hidden; box-shadow:0 8px 24px rgba(0,0,0,.10); }}
div[data-testid="stMetric"] {{ background:var(--surface); border-radius:18px; padding:1rem; border:1px solid var(--border); }}
h2,h3 {{ color:var(--g700); font-weight:700 !important; }}
hr {{ border-color:var(--g200) !important; }}
.footer {{ text-align:center; color:var(--muted); font-size:.82rem; margin-top:2.2rem; }}
::-webkit-scrollbar {{ width:9px; }} ::-webkit-scrollbar-thumb {{ background:var(--g300); border-radius:9px; }}
</style>
"""
