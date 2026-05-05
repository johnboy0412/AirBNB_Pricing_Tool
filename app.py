import streamlit as st
import pandas as pd
import numpy as np
import joblib
import altair as alt
import anthropic
import shap
import os
from dotenv import load_dotenv
load_dotenv()

# ── Configuration & Styling ────────────────────────────────────────────────────
st.set_page_config(page_title="National Airbnb Optimizer", page_icon="🏠", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main-header { font-family: 'Inter', sans-serif; color: #ff5a5f; font-weight: 800; text-align: center; padding-bottom: 5px; }
    .sub-header { text-align: center; color: #484848; font-size: 1.1rem; margin-bottom: 40px; }
    .price-box {
        background: linear-gradient(135deg, #ff5a5f 0%, #ff385c 100%);
        color: white; padding: 30px; border-radius: 15px; text-align: center;
        box-shadow: 0 10px 20px rgba(255, 90, 95, 0.2); margin-top: 10px; margin-bottom: 20px;
    }
    .price-text { font-size: 4rem; font-weight: 900; margin: 0; line-height: 1; }
    .price-range { font-size: 1.1rem; opacity: 0.9; margin-top: 8px; }
    .feature-tag {
        display: inline-block; background-color: #00a699; color: white;
        padding: 6px 16px; border-radius: 20px; margin: 4px;
        font-size: 0.9rem; font-weight: bold; box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .ai-box {
        background: #f0f9f8; border-left: 4px solid #00a699;
        padding: 20px; border-radius: 8px; margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        preprocessor      = joblib.load('preprocessor.pkl')
        model             = joblib.load('airbnb_model.pkl')
        city_neighborhoods = joblib.load('city_neighborhoods.pkl')
        importance_df     = joblib.load('feature_importances.pkl')
        feature_names     = joblib.load('feature_names.pkl')
        neighborhood_stats = joblib.load('neighborhood_stats.pkl')
        return preprocessor, model, city_neighborhoods, importance_df, feature_names, neighborhood_stats
    except Exception as e:
        st.error(f"Model files not found. Run train_model.py first. Error: {e}")
        st.stop()

preprocessor, ml_model, city_neighborhoods, importance_df, feature_names, neighborhood_stats = load_models()


# ── Regex Feature Extractor (matches training logic exactly) ───────────────────
def extract_features_from_text(text: str) -> dict:
    t = text.lower()
    return {
        'has_pool':    1 if re.search(r'\bpool\b', t) else 0,
        'has_hot_tub': 1 if re.search(r'hot tub|jacuzzi', t) else 0,
        'has_view':    1 if re.search(r'\bview\b', t) else 0,
        'is_luxury':   1 if re.search(r'luxury|premium|high-end', t) else 0,
    }

import re  # needed by extract_features_from_text


# ── Build input row for the model ──────────────────────────────────────────────
def build_input(city, neighborhood, room_type, bedrooms, baths, accommodates,
                review_score, num_reviews, is_superhost, instant_bookable,
                amenity_count, features: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        'city': city,
        'neighbourhood_cleansed': neighborhood,
        'room_type': room_type,
        'data_year': '2025',
        'bedrooms': bedrooms,
        'baths': baths,
        'accommodates': accommodates,
        'review_score': review_score,
        'num_reviews': num_reviews,
        'is_superhost': is_superhost,
        'instant_bookable': instant_bookable,
        'amenity_count': amenity_count,
        **features,
    }])


# ── Claude explanation ─────────────────────────────────────────────────────────
def get_ai_explanation(city, neighborhood, room_type, bedrooms, baths,
                       final_price, base_price, price_boost, detected_features,
                       comp_low, comp_high, shap_top) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "_AI explanation unavailable — add your ANTHROPIC_API_KEY to the environment or Streamlit secrets._"

    shap_lines = "\n".join([f"  - {name}: {val:+.1f}" for name, val in shap_top])

    prompt = f"""You are a helpful real estate pricing advisor for Airbnb hosts.

A homeowner just used an ML model to price their listing. Provide a concise, friendly 3–4 sentence explanation of their result. Cover:
1. Why the model landed on this price given their location and property details.
2. How their description affected the price (if at all).
3. One specific, actionable tip to increase their nightly rate.

Keep the tone warm and practical. Do not repeat the raw numbers back mechanically — interpret them.

Listing details:
- City: {city}, Neighborhood: {neighborhood}
- Room type: {room_type}, Bedrooms: {bedrooms}, Bathrooms: {baths}
- Predicted price: ${final_price:.0f}/night (base: ${base_price:.0f}, description boost: ${price_boost:.0f})
- Similar listings in this neighborhood: ${comp_low:.0f}–${comp_high:.0f}/night
- Premium features detected in description: {detected_features if detected_features else 'none'}
- Top factors driving this specific prediction (SHAP):
{shap_lines}
"""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"_AI explanation unavailable: {e}_"


# ── SHAP helper ────────────────────────────────────────────────────────────────
@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)

def compute_shap(explainer, X_transformed, feature_names_list, top_n=8):
    shap_vals = explainer.shap_values(X_transformed)
    pairs = list(zip(feature_names_list, shap_vals[0]))
    pairs.sort(key=lambda x: abs(x[1]), reverse=True)

    def clean(name):
        name = name.replace('cat__', '').replace('remainder__', '')
        for prefix, label in [
            ('room_type_', 'Room: '), ('city_', 'City: '),
            ('neighbourhood_cleansed_', 'Neighborhood: '), ('data_year_', 'Year: ')
        ]:
            if name.startswith(prefix):
                return label + name[len(prefix):]
        return name.replace('_', ' ').title()

    return [(clean(n), v) for n, v in pairs[:top_n]]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/Airbnb_Logo_B%C3%A9lo.svg", width=150)
    st.markdown("### Location")

    available_cities = sorted(list(city_neighborhoods.keys()))
    default_city_idx = available_cities.index('New York City') if 'New York City' in available_cities else 0
    city = st.selectbox("City", available_cities, index=default_city_idx)

    valid_neighborhoods = city_neighborhoods[city]
    neighborhood = st.selectbox("Neighborhood", valid_neighborhoods)

    st.markdown("### Property Basics")
    room_type     = st.selectbox("Room Type", ["Entire home/apt", "Private room", "Shared room", "Hotel room"])
    bedrooms      = st.slider("Bedrooms",   min_value=0,   max_value=10,  value=1)
    baths         = st.slider("Bathrooms",  min_value=0.5, max_value=10.0, step=0.5, value=1.0)
    accommodates  = st.slider("Max Guests", min_value=1,   max_value=16,  value=2)

    st.markdown("### Host Profile")
    review_score    = st.slider("Avg Review Score", min_value=1.0, max_value=5.0, step=0.1, value=4.7)
    num_reviews     = st.slider("Number of Reviews", min_value=0, max_value=500, value=20)
    amenity_count   = st.slider("Number of Amenities", min_value=0, max_value=80, value=25)
    is_superhost    = st.checkbox("Superhost", value=False)
    instant_bookable = st.checkbox("Instant Bookable", value=False)


# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown("<h1 class='main-header'>🏠 National Airbnb Pricing Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Powered by Machine Learning & AI across 200,000+ US Listings</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Price Optimizer", "ML Transparency Dashboard"])

with tab1:
    st.markdown("### Listing Description")
    description = st.text_area(
        "Paste or write your Airbnb listing description. The model scans it for premium features that affect price.",
        value="Welcome to our beautiful cozy apartment! We just added a brand new luxury finish to the kitchen. Step out onto the private balcony for an amazing scenic view, or head downstairs to relax in the hot tub.",
        height=120
    )

    analyze_btn = st.button("Analyze & Price Listing", type="primary", use_container_width=True)

    if analyze_btn:
        st.divider()

        # 1. Extract text features (regex — same as training)
        text_features = extract_features_from_text(description)
        detected = []
        if text_features['has_pool']:    detected.append("Pool")
        if text_features['has_hot_tub']: detected.append("Hot Tub")
        if text_features['has_view']:    detected.append("Scenic View")
        if text_features['is_luxury']:   detected.append("Luxury")

        host_flags = {
            'is_superhost':     int(is_superhost),
            'instant_bookable': int(instant_bookable),
        }

        # 2. Base price (no text premium features)
        base_row   = build_input(city, neighborhood, room_type, bedrooms, baths, accommodates,
                                  review_score, num_reviews, host_flags['is_superhost'],
                                  host_flags['instant_bookable'], amenity_count,
                                  {'has_pool': 0, 'has_hot_tub': 0, 'has_view': 0, 'is_luxury': 0})
        base_trans = preprocessor.transform(base_row)
        base_price = np.expm1(ml_model.predict(base_trans)[0])

        # 3. Final price (with detected text features)
        final_row   = build_input(city, neighborhood, room_type, bedrooms, baths, accommodates,
                                   review_score, num_reviews, host_flags['is_superhost'],
                                   host_flags['instant_bookable'], amenity_count, text_features)
        final_trans = preprocessor.transform(final_row)
        final_price = np.expm1(ml_model.predict(final_trans)[0])

        price_boost = max(final_price - base_price, 0)

        # 4. Neighborhood comps
        mask = (
            (neighborhood_stats['city'] == city) &
            (neighborhood_stats['neighbourhood_cleansed'] == neighborhood)
        )
        comp_row  = neighborhood_stats[mask]
        comp_low  = float(comp_row['p25'].iloc[0])  if not comp_row.empty else base_price * 0.75
        comp_high = float(comp_row['p75'].iloc[0])  if not comp_row.empty else base_price * 1.25
        comp_med  = float(comp_row['median'].iloc[0]) if not comp_row.empty else base_price

        # 5. SHAP
        explainer  = get_shap_explainer(ml_model)
        shap_top   = compute_shap(explainer, final_trans, feature_names)

        # 6. AI explanation
        with st.spinner("Generating AI explanation..."):
            explanation = get_ai_explanation(
                city, neighborhood, room_type, bedrooms, baths,
                final_price, base_price, price_boost, detected,
                comp_low, comp_high, shap_top
            )

        # ── Output layout ──────────────────────────────────────────────────────
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"""
            <div class="price-box">
                <p style="margin:0;font-size:1.2rem;opacity:0.9;">Recommended Nightly Price — {city}</p>
                <p class="price-text">${final_price:.0f}</p>
                <p class="price-range">Similar listings in {neighborhood}: ${comp_low:.0f} – ${comp_high:.0f}/night</p>
            </div>
            """, unsafe_allow_html=True)

            if detected:
                tags = " ".join([f"<span class='feature-tag'>{d}</span>" for d in detected])
                st.markdown(f"**Detected in your description:** {tags}", unsafe_allow_html=True)
            else:
                st.info("No premium keywords detected in description.")

            if price_boost > 0:
                st.success(f"Your description added **${price_boost:.0f}/night** above the base estimate.")

            st.markdown(f'<div class="ai-box"><b>AI Advisor</b><br><br>{explanation}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown("#### Price Breakdown")
            chart_data = pd.DataFrame({
                "Component": ["Base (Location/Size/Profile)", "Description Boost"],
                "Price ($)": [base_price, price_boost],
                "Color": ["#484848", "#00a699"]
            })
            bar = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X('Component', sort=None, axis=alt.Axis(labelAngle=0, title="")),
                y=alt.Y('Price ($)', title="Nightly Value ($)"),
                color=alt.Color('Color', scale=None, legend=None),
                tooltip=['Component', 'Price ($)']
            ).properties(height=220)
            st.altair_chart(bar, use_container_width=True)

            st.markdown("#### What's Driving Your Price (SHAP)")
            st.caption("Positive = pushes price up. Negative = pushes price down.")
            shap_df = pd.DataFrame(shap_top, columns=['Feature', 'SHAP Value'])
            shap_df['Color'] = shap_df['SHAP Value'].apply(lambda v: '#00a699' if v > 0 else '#ff5a5f')
            shap_chart = alt.Chart(shap_df).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
                x=alt.X('SHAP Value:Q', title="Impact on log-price"),
                y=alt.Y('Feature:N', sort=alt.EncodingSortField(field='SHAP Value', order='descending'), title=""),
                color=alt.Color('Color:N', scale=None, legend=None),
                tooltip=['Feature', 'SHAP Value']
            ).properties(height=280)
            st.altair_chart(shap_chart, use_container_width=True)

            st.markdown("#### Neighborhood Comps")
            comp_df = pd.DataFrame({
                'Metric': ['25th Percentile', 'Median', '75th Percentile', 'Your Price'],
                'Price': [comp_low, comp_med, comp_high, final_price],
                'Color': ['#ccc', '#484848', '#ccc', '#ff5a5f']
            })
            comp_chart = alt.Chart(comp_df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X('Metric', sort=None, axis=alt.Axis(labelAngle=-20, title="")),
                y=alt.Y('Price', title="Nightly Price ($)"),
                color=alt.Color('Color:N', scale=None, legend=None),
                tooltip=['Metric', 'Price']
            ).properties(height=220)
            st.altair_chart(comp_chart, use_container_width=True)


with tab2:
    st.markdown("### Global Feature Importance")
    st.markdown("These are the features that explain the most variance in nightly price across all cities and listings in the training data.")

    importance_chart = alt.Chart(importance_df).mark_bar(
        color='#ff5a5f', cornerRadiusTopRight=3, cornerRadiusBottomRight=3
    ).encode(
        x=alt.X('Importance:Q', title="Relative Importance Score"),
        y=alt.Y('Feature:N', sort='-x', title="Property Feature"),
        tooltip=['Feature', 'Importance']
    ).properties(height=550)

    st.altair_chart(importance_chart, use_container_width=True)
    st.info("These global importances show which features matter most on average. The SHAP chart in the Price Optimizer tab shows the same breakdown personalized to your specific listing.")
