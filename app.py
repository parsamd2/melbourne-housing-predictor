import streamlit as st
import pandas as pd
import numpy as np
import joblib
import urllib.request
import json

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Melbourne Housing Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load model ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("xgboost_pipeline.pkl")

model = load_model()

# ── Load schools from Victorian Government API ────────────────────
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

# ── Fixed lookup tables ───────────────────────────────────────────
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

# Property types that must have land size
NEEDS_LAND = {"House", "Townhouse", "Villa", "Vacant land", "Block of Units"}

# ── Header ────────────────────────────────────────────────────────
st.title("🏠 Melbourne Housing Price Predictor")
st.markdown(
    "Predict the **sold price** of a property in **Richmond**, "
    "**Hawthorn**, or **Box Hill** based on its features."
)
st.divider()

# ── Inputs ────────────────────────────────────────────────────────
st.subheader("Property Details")
col1, col2 = st.columns(2)

with col1:
    suburb = st.selectbox("Suburb", ["Richmond", "Hawthorn", "Box Hill"])
    ptype  = st.selectbox("Property Type", [
        "House",
        "Apartment / Unit / Flat",
        "Townhouse",
        "Villa",
        "Studio",
        "Block of Units",
        "New Apartments / Off the Plan",
        "Vacant land",
    ])
    beds  = st.slider("Bedrooms",  min_value=1, max_value=6, value=2)
    baths = st.slider("Bathrooms", min_value=1, max_value=5, value=1)

with col2:
    cars = st.slider("Car Spaces", min_value=0, max_value=4, value=1)

    # Land size: required for houses/townhouses, optional for apartments
    if ptype in NEEDS_LAND:
        land = st.number_input(
            "Land Size (sqm) — required for this property type",
            min_value=1, max_value=5000, value=300, step=10
        )
    else:
        land = st.number_input(
            "Land Size (sqm) — enter 0 for apartments/units",
            min_value=0, max_value=2000, value=0, step=10
        )

    method = st.radio(
        "Sale Method", ["Private Treaty", "Auction"], horizontal=True
    )
    month  = st.selectbox(
        "Month of Sale",
        list(MONTHS.keys()),
        format_func=lambda x: MONTHS[x],
        index=3
    )

st.divider()

# ── Predict button ────────────────────────────────────────────────
if st.button("💰 Predict Price", type="primary", use_container_width=True):

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

    # Sanity checks
    if price < 150000:
        st.warning("⚠️ Prediction seems unusually low — please check your inputs.")
    elif price > 10000000:
        st.warning("⚠️ Prediction seems unusually high — please check your inputs.")

    # Result — use st.metric so fonts are consistent
    st.success(f"### Estimated Sold Price:  ${price:,.0f}")

    c_low, c_dash, c_high = st.columns([5, 1, 5])
    with c_low:
        st.metric("Lower estimate (−10%)", f"${low:,.0f}")
    with c_dash:
        st.markdown(
            "<div style='text-align:center;padding-top:28px;font-size:22px;'></div>",
            unsafe_allow_html=True
        )
    with c_high:
        st.metric("Upper estimate (+10%)", f"${high:,.0f}")

    st.divider()

    # Input summary
    st.subheader("Input Summary")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Suburb",        suburb)
        st.metric("Property Type", ptype.split("/")[0].strip())
        st.metric("Bedrooms",      beds)
        st.metric("Bathrooms",     baths)

    with c2:
        st.metric("Car Spaces",    cars)
        st.metric("Land Size",     f"{land} sqm" if land > 0 else "N/A")
        st.metric("Sale Method",   method)
        st.metric("Month of Sale", MONTHS[month])

    with c3:
        st.metric("Dist. to CBD",   f"{CBD_KM[suburb]} km")
        st.metric("Schools Nearby", schools_n)
        st.metric("Has Land",       "Yes" if has_land else "No")
        st.metric("Is Auction",     "Yes" if is_auction else "No")