import os
import streamlit as st
import pandas as pd
import joblib
import folium

from streamlit_folium import st_folium
from geopy.distance import geodesic

try:
    from optimizer import simulate_factories, recommend_factory
except Exception as e:
    st.error(f"❌ optimizer.py missing or error: {e}")
    st.stop()
st.set_page_config(page_title="Factory Optimization AI", layout="wide")
st.title("🏭 Factory Reallocation & Shipping Optimization System")

if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.best = None

@st.cache_data
def load_data():
    try:
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, "data", "nassau_dataset.csv")
        return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"❌ Dataset load failed: {e}")
        return None


@st.cache_resource
def load_encoders():
    try:
        base_path = os.path.dirname(__file__)
        file_path = os.path.join(base_path, "models", "encoders.pkl")
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"❌ Encoder load failed: {e}")
        return None


df = load_data()
encoders = load_encoders()

if df is None or encoders is None:
    st.stop()

st.sidebar.header("Simulation Inputs")

product = st.sidebar.selectbox("Product", df["Product Name"].dropna().unique())
region = st.sidebar.selectbox("Region", df["Region"].dropna().unique())
ship_mode = st.sidebar.selectbox("Ship Mode", df["Ship Mode"].dropna().unique())

units = st.sidebar.slider("Units", 1, 100, 10)
cost = st.sidebar.number_input("Cost", 1.0, 1000.0, 50.0)

filtered = df[df["Product Name"] == product]

if filtered.empty:
    st.error("❌ No matching product found")
    st.stop()

division = filtered["Division"].iloc[0]

def safe_encode(encoders, column, value):
    try:
        return encoders[column].transform([value])[0]
    except:
        return 0  
input_df = pd.DataFrame({
    "Ship Mode": [safe_encode(encoders, "Ship Mode", ship_mode)],
    "Region": [safe_encode(encoders, "Region", region)],
    "Division": [safe_encode(encoders, "Division", division)],
    "Product Name": [safe_encode(encoders, "Product Name", product)],
    "Units": [units],
    "Cost": [cost]
})

factory_locations = pd.DataFrame({
    "Factory": [
        "Lot's O' Nuts",
        "Wicked Choccy's",
        "Sugar Shack",
        "Secret Factory",
        "The Other Factory"
    ],
    "Latitude": [32.881893, 32.076176, 48.11914, 41.446333, 35.1175],
    "Longitude": [-111.768036, -81.088371, -96.18115, -90.565487, -89.971107]
})

customer_location = (40.7128, -74.0060)

if st.button("Run Optimization"):
    try:
        results = simulate_factories(input_df)

        if not isinstance(results, (list, dict)):
            st.error("❌ Invalid simulation output")
            st.stop()

        results_df = pd.DataFrame(results)

        best = recommend_factory(results)

        if not isinstance(best, dict) or "Factory" not in best:
            st.error("❌ Invalid recommendation output")
            st.stop()

        st.session_state.results = results_df
        st.session_state.best = best

    except Exception as e:
        st.error(f"❌ Simulation failed: {e}")
        st.stop()

if st.session_state.results is not None:

    st.subheader("📊 Predicted Lead Time by Factory")
    st.dataframe(st.session_state.results)

    best = st.session_state.best

    st.success(
        f"✅ Recommended Factory: {best.get('Factory', 'N/A')} | "
        f"Lead Time: {best.get('Predicted Lead Time', 0):.2f} days"
    )

    try:
        st.bar_chart(st.session_state.results.set_index("Factory"))
    except:
        st.warning("⚠️ Could not render chart")

    st.subheader("🗺 Logistics Map Simulation")

    m = folium.Map(location=[39, -95], zoom_start=4)

    
    folium.Marker(
        customer_location,
        popup="Customer",
        icon=folium.Icon(color="red")
    ).add_to(m)

    for _, row in factory_locations.iterrows():

        factory_point = (row["Latitude"], row["Longitude"])

        color = "green" if row["Factory"] == best["Factory"] else "blue"

        folium.Marker(
            factory_point,
            popup=row["Factory"],
            icon=folium.Icon(color=color, icon="industry")
        ).add_to(m)

        folium.PolyLine(
            [factory_point, customer_location],
            color=color,
            weight=2
        ).add_to(m)

        try:
            distance = geodesic(factory_point, customer_location).km
        except:
            distance = 0

        folium.Marker(
            [
                (factory_point[0] + customer_location[0]) / 2,
                (factory_point[1] + customer_location[1]) / 2
            ],
            popup=f"{distance:.0f} km"
        ).add_to(m)

    st_folium(m, width=900, height=500)
