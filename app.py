import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import json

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Suburb Price Estimator",
    page_icon="🏙",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #F7F5F2;
    color: #1A1A1A;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 680px; }

/* Hero title */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    line-height: 1.15;
    color: #1A1A1A;
    margin-bottom: 0.25rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.95rem;
    color: #777;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 0.3px;
}

/* Section label */
.section-label {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.75rem;
    margin-top: 1.5rem;
}

/* Result card */
.result-card {
    background: #1A1A1A;
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
    text-align: center;
}
.result-label {
    font-size: 0.72rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #888;
    font-weight: 500;
    margin-bottom: 0.5rem;
}
.result-price {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    color: #F7F5F2;
    letter-spacing: -1px;
    line-height: 1;
}
.result-range {
    font-size: 0.82rem;
    color: #666;
    margin-top: 0.75rem;
    font-weight: 300;
}
.result-range span {
    color: #aaa;
    font-weight: 400;
}

/* Info badge */
.info-badge {
    display: inline-block;
    background: #EDE9E3;
    border-radius: 6px;
    padding: 0.35rem 0.75rem;
    font-size: 0.8rem;
    color: #555;
    margin-top: 0.25rem;
}

/* Summary grid */
.summary-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 0.75rem;
    margin-top: 1rem;
}
.summary-item {
    background: #EDEAE4;
    border-radius: 10px;
    padding: 0.85rem 1rem;
}
.summary-key {
    font-size: 0.65rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.2rem;
}
.summary-val {
    font-size: 0.95rem;
    font-weight: 500;
    color: #1A1A1A;
}

/* Warning */
.warn-box {
    background: #FFF4E5;
    border-left: 3px solid #F5A623;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    font-size: 0.85rem;
    color: #8A6200;
    margin: 1rem 0;
}

/* Divider */
.thin-line {
    border: none;
    border-top: 1px solid #E5E2DC;
    margin: 1.5rem 0;
}

/* Streamlit widget overrides */
.stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    color: #555 !important;
    text-transform: uppercase !important;
}
.stButton > button {
    background-color: #1A1A1A !important;
    color: #F7F5F2 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    padding: 0.65rem 2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
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

# ── Hero ──────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Property Price<br><i>Estimator</i></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Melbourne · Richmond · Hawthorn · Box Hill</div>', unsafe_allow_html=True)

# ── Location ──────────────────────────────────────────────────────
st.markdown('<div class="section-label">Location & Type</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    suburb = st.selectbox("Suburb", list(COORDS.keys()), label_visibility="collapsed")
with col2:
    ptype  = st.selectbox("Property Type", list(PTYPE_RULES.keys()), label_visibility="collapsed")

rules = PTYPE_RULES[ptype]

# ── Property details ──────────────────────────────────────────────
st.markdown('<div class="section-label">Property Details</div>', unsafe_allow_html=True)

if rules["fixed"]:
    beds  = 1
    baths = 1
    st.markdown('<div class="info-badge">🛏 Studio — 1 bed · 1 bath fixed</div>', unsafe_allow_html=True)
    st.write("")
else:
    c1, c2 = st.columns(2)
    with c1:
        beds  = st.slider("Bedrooms",  min_value=rules["min_beds"], max_value=rules["max_beds"],  value=min(2, rules["max_beds"]))
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
        st.markdown('<div style="margin-top:1.8rem"><div class="info-badge">🏢 No land — apartment/studio</div></div>', unsafe_allow_html=True)

# ── Sale details ──────────────────────────────────────────────────
st.markdown('<div class="section-label">Sale Details</div>', unsafe_allow_html=True)
c5, c6 = st.columns(2)
with c5:
    method = st.radio("Sale Method", ["Private Treaty", "Auction"], horizontal=True)
with c6:
    month  = st.selectbox("Month of Sale", list(MONTHS.keys()),
                          format_func=lambda x: MONTHS[x], index=3)

st.write("")

# ── Predict ───────────────────────────────────────────────────────
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
                <span>${low:,.0f}</span> &nbsp;—&nbsp; <span>${high:,.0f}</span>
                &nbsp;&nbsp;·&nbsp;&nbsp; ±10% typical range
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