import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import json

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Property Price Estimator",
    page_icon="✦",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Inter:wght@300;400;500;600&display=swap');

/* Global */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #0F1115 !important;
    color: #EDE8E0;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2.5rem;
    padding-bottom: 3rem;
    max-width: 720px;
}

/* ── HERO ───────────────────────────────────────────── */
.hero-wrap {
    text-align: center;
    margin-bottom: 3rem;
    padding-top: 1rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem;
    line-height: 1;
    color: #EDE8E0;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
    font-weight: 500;
}
.hero-title em {
    color: #D97757;
    font-style: italic;
    font-weight: 400;
}
.hero-sub {
    font-family: 'Inter', sans-serif;
    color: #8A8378;
    font-weight: 400;
    text-transform: uppercase;
}
.city-main {
    font-size: 1.1rem;
    color: #EDE8E0;
    font-weight: 600;
    letter-spacing: 5px;
}
.city-sub {
    font-size: 0.72rem;
    color: #8A8378;
    letter-spacing: 3px;
    margin-left: 8px;
}
.hero-divider {
    width: 60px;
    height: 1px;
    background: #D97757;
    margin: 1.25rem auto 0 auto;
    opacity: 0.6;
}

/* ── SECTION LABELS ─────────────────────────────────── */
.section-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 500;
    color: #EDE8E0;
    margin-top: 2.25rem;
    margin-bottom: 1rem;
    letter-spacing: -0.3px;
}
.section-label em {
    color: #D97757;
    font-style: italic;
}

/* ── INFO BADGE ─────────────────────────────────────── */
.info-badge {
    display: inline-block;
    background: rgba(217, 119, 87, 0.08);
    border: 1px solid rgba(217, 119, 87, 0.3);
    border-radius: 4px;
    padding: 0.5rem 0.9rem;
    font-size: 0.82rem;
    color: #D97757;
    letter-spacing: 0.3px;
}

/* ── RESULT CARD ────────────────────────────────────── */
.result-card {
    background: linear-gradient(145deg, #1A1D24 0%, #14161B 100%);
    border: 1px solid rgba(217, 119, 87, 0.2);
    border-radius: 4px;
    padding: 2.5rem 2rem;
    margin: 2rem 0;
    text-align: center;
    position: relative;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 40px;
    height: 1px;
    background: #D97757;
}
.result-label {
    font-size: 0.7rem;
    letter-spacing: 4px;
    text-transform: uppercase;
    color: #8A8378;
    font-weight: 500;
    margin-bottom: 1rem;
}
.result-price {
    font-family: 'Inter', sans-serif;
    font-size: 3.2rem;
    color: #EDE8E0;
    letter-spacing: -1.5px;
    line-height: 1;
    font-weight: 500;
}
.result-range {
    font-size: 0.8rem;
    color: #6B6358;
    margin-top: 1.25rem;
    font-weight: 400;
    letter-spacing: 0.5px;
}
.result-range b {
    color: #B8AEA0;
    font-weight: 500;
}

/* ── WARNING ────────────────────────────────────────── */
.warn-box {
    background: rgba(245, 166, 35, 0.08);
    border-left: 2px solid #F5A623;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    color: #F5A623;
    margin: 1.5rem 0;
    letter-spacing: 0.2px;
}

/* ── DIVIDER ────────────────────────────────────────── */
.thin-line {
    border: none;
    border-top: 1px solid rgba(237, 232, 224, 0.08);
    margin: 2rem 0;
}

/* ── SUMMARY GRID ───────────────────────────────────── */
.summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.75rem;
    margin-top: 0.5rem;
}
.summary-item {
    background: rgba(237, 232, 224, 0.04);
    border: 1px solid rgba(237, 232, 224, 0.06);
    border-radius: 3px;
    padding: 0.85rem 1rem;
}
.summary-key {
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #6B6358;
    margin-bottom: 0.3rem;
}
.summary-val {
    font-size: 0.95rem;
    font-weight: 500;
    color: #EDE8E0;
}

/* ── STREAMLIT WIDGETS ──────────────────────────────── */
.stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    letter-spacing: 2px !important;
    color: #8A8378 !important;
    text-transform: uppercase !important;
}

/* Selectbox text */
.stSelectbox > div > div, .stNumberInput > div > div > input {
    background-color: rgba(237, 232, 224, 0.04) !important;
    color: #EDE8E0 !important;
    border: 1px solid rgba(237, 232, 224, 0.1) !important;
    border-radius: 3px !important;
}

/* Sliders */
.stSlider [data-baseweb="slider"] > div > div { background: #D97757 !important; }
.stSlider [role="slider"] { background: #D97757 !important; border: none !important; }

/* Radio */
.stRadio > div { gap: 0.5rem !important; }

/* ── PRIMARY BUTTON ─────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #D97757 0%, #C96644 100%) !important;
    color: #0F1115 !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    letter-spacing: 4px !important;
    padding: 1.1rem 2rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(217, 119, 87, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 25px rgba(217, 119, 87, 0.4) !important;
    background: linear-gradient(135deg, #E5825F 0%, #D17252 100%) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}


/* Tick bar labels at slider ends */
[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"] {
    background: transparent !important;
    background-color: transparent !important;
    color: #6B6358 !important;
    font-size: 0.7rem !important;
    box-shadow: none !important;
    border: none !important;
}

/* ── KILL ORANGE BACKGROUND ON ALL SLIDER VALUE TOOLTIPS ── */
.stSlider [data-baseweb="tooltip"],
.stSlider [data-baseweb="tooltip"] > div,
.stSlider [data-baseweb="tooltip"] * {
    background: transparent !important;
    background-color: transparent !important;
    box-shadow: none !important;
    border: none !important;
    color: #EDE8E0 !important;
}

[data-testid="stSliderThumbValue"] {
    background: transparent !important;
    background-color: transparent !important;
    color: #EDE8E0 !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stSliderThumbValue"] * {
    background: transparent !important;
    background-color: transparent !important;
    color: #EDE8E0 !important;
    box-shadow: none !important;
}

/* Tick labels at ends */
[data-testid="stTickBarMin"],
[data-testid="stTickBarMax"] {
    background: transparent !important;
    background-color: transparent !important;
    color: #6B6358 !important;
    font-size: 0.7rem !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stTickBarMin"] *,
[data-testid="stTickBarMax"] * {
    background: transparent !important;
    background-color: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgboost_pipeline.pkl")

model = load_model()

# ── Load schools ──────────────────────────────────────────────────
@st.cache_data
def load_schools():
    url  = ("https://discover.data.vic.gov.au/api/3/action/datastore_search"
            "?resource_id=d26bf015-a1e5-48dd-a1d6-8edd4b0a511b&limit=10000")
    data    = json.loads(urllib.request.urlopen(url).read())
    schools = pd.DataFrame(data["result"]["records"])
    schools = schools[schools["School_Status"] == "O"].copy()
    schools["X"] = pd.to_numeric(schools["X"], errors="coerce")
    schools["Y"] = pd.to_numeric(schools["Y"], errors="coerce")
    return schools.dropna(subset=["X", "Y"])

schools     = load_schools()
school_lats = schools["Y"].values
school_lons = schools["X"].values

def count_schools(lat, lon, r=2):
    dlat = np.radians(school_lats - lat)
    dlon = np.radians(school_lons - lon)
    a    = (np.sin(dlat/2)**2 +
            np.cos(np.radians(lat)) * np.cos(np.radians(school_lats)) *
            np.sin(dlon/2)**2)
    return int((6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a)) <= r).sum())

# ── Constants ─────────────────────────────────────────────────────
COORDS = {
    "Richmond": (-37.823, 145.000),
    "Hawthorn": (-37.823, 145.033),
    "Box Hill": (-37.819, 145.124),
}
CBD_KM = {"Richmond": 3.5, "Hawthorn": 6.0, "Box Hill": 14.0}
MONTHS = {
    1:"January",  2:"February", 3:"March",    4:"April",
    5:"May",      6:"June",     7:"July",      8:"August",
    9:"September",10:"October",11:"November", 12:"December"
}
PTYPE_RULES = {
    "House":                   dict(min_beds=1, max_beds=6, max_baths=5, max_cars=4, needs_land=True,  fixed=False),
    "Townhouse":               dict(min_beds=1, max_beds=5, max_baths=4, max_cars=3, needs_land=True,  fixed=False),
    "Villa":                   dict(min_beds=1, max_beds=4, max_baths=3, max_cars=2, needs_land=True,  fixed=False),
    "Apartment / Unit / Flat": dict(min_beds=1, max_beds=4, max_baths=3, max_cars=2, needs_land=False, fixed=False),
    "Studio":                  dict(min_beds=1, max_beds=1, max_baths=1, max_cars=1, needs_land=False, fixed=True),
}

# ── HERO ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <div class="hero-title">Property Price <em>Estimator</em></div>
    <div class="hero-divider"></div>
    <div class="hero-sub" style="margin-top: 1.25rem"><span class="city-main">Melbourne</span> <span class="city-sub">· Richmond · Hawthorn · Box Hill</span></div>
</div>
""", unsafe_allow_html=True)

# ── LOCATION & TYPE ───────────────────────────────────────────────
st.markdown('<div class="section-label">Location & Type</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    suburb = st.selectbox("Suburb", list(COORDS.keys()), label_visibility="collapsed")
with col2:
    ptype  = st.selectbox("Property Type", list(PTYPE_RULES.keys()), label_visibility="collapsed")

rules = PTYPE_RULES[ptype]

# ── PROPERTY DETAILS ──────────────────────────────────────────────
st.markdown('<div class="section-label">Property Details</div>', unsafe_allow_html=True)

if rules["fixed"]:
    beds  = 1
    baths = 1
    st.markdown('<div class="info-badge">✦ Studio · 1 bedroom · 1 bathroom</div>', unsafe_allow_html=True)
    st.write("")
else:
    c1, c2 = st.columns(2)
    with c1:
        beds  = st.slider("Bedrooms",  min_value=rules["min_beds"], max_value=rules["max_beds"], value=min(2, rules["max_beds"]))
    with c2:
        baths = st.slider("Bathrooms", min_value=1, max_value=rules["max_baths"], value=1)

c3, c4 = st.columns(2)
with c3:
    cars = st.slider("Car Spaces", min_value=0, max_value=rules["max_cars"], value=min(1, rules["max_cars"]))
with c4:
    if rules["needs_land"]:
        land = st.number_input("Land Size (sqm)", min_value=1, max_value=5000, value=300, step=10)
    else:
        land = 0
        st.markdown('<div style="margin-top:1.6rem"><div class="info-badge">✦ No land · apartment / studio</div></div>', unsafe_allow_html=True)

# ── SALE DETAILS ──────────────────────────────────────────────────
st.markdown('<div class="section-label">Sale Details</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    method = st.radio("Sale Method", ["Private Treaty", "Auction"], horizontal=True)
with c6:
    month  = st.selectbox("Month of Sale", list(MONTHS.keys()),
                          format_func=lambda x: MONTHS[x], index=3)

st.write("")
st.write("")

# ── PREDICT ───────────────────────────────────────────────────────
if st.button("ESTIMATE PRICE", use_container_width=True):

    lat, lon   = COORDS[suburb]
    schools_n  = count_schools(lat, lon)
    is_auction = 1 if method == "Auction" else 0
    has_land   = 1 if land > 0 else 0

    row = pd.DataFrame([{
        "bedrooms":       beds,
        "bathrooms":      baths,
        "car_spaces":     cars,
        "land_size_sqm":  float(land),
        "month_sold":     month,
        "year_sold":      2025,
        "dist_to_cbd_km": CBD_KM[suburb],
        "is_auction":     is_auction,
        "has_land":       has_land,
        "schools_nearby": schools_n,
        "suburb":         suburb,
        "property_type":  ptype,
    }])

    price = model.predict(row)[0]
    low   = price * 0.90
    high  = price * 1.10

    if price < 150000 or price > 10000000:
        st.markdown(
            f'<div class="warn-box">⚠ Prediction of ${price:,.0f} seems outside normal range — please review your inputs.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Sold Price</div>
            <div class="result-price">${price:,.0f}</div>
            <div class="result-range">
                <b>${low:,.0f}</b> &nbsp; — &nbsp; <b>${high:,.0f}</b>
                <br><span style="font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase;">±10% typical range</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<hr class="thin-line">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)

    items = [
        ("Suburb",        suburb),
        ("Type",          ptype.split("/")[0].strip()),
        ("Bedrooms",      beds),
        ("Bathrooms",     baths),
        ("Car Spaces",    cars),
        ("Land Size",     f"{land} sqm" if land > 0 else "N/A"),
        ("Sale Method",   method),
        ("Month",         MONTHS[month]),
        ("Dist. to CBD",  f"{CBD_KM[suburb]} km"),
        ("Schools Nearby",schools_n),
        ("Has Land",      "Yes" if has_land else "No"),
        ("Is Auction",    "Yes" if is_auction else "No"),
    ]

    grid_html = '<div class="summary-grid">'
    for key, val in items:
        grid_html += f'''
        <div class="summary-item">
            <div class="summary-key">{key}</div>
            <div class="summary-val">{val}</div>
        </div>'''
    grid_html += '</div>'
    st.markdown(grid_html, unsafe_allow_html=True)