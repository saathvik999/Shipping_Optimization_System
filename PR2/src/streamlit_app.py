import streamlit as st
import pandas as pd
import joblib
import folium

from streamlit_folium import st_folium
from geopy.distance import geodesic
from optimizer import simulate_factories, recommend_factory

st.set_page_config(page_title="Factory Optimization AI",layout="wide")

st.title("Factory Reallocation & Shipping Optimization System")

# Initialize session state
if 'results' not in st.session_state:
    st.session_state.results = None
    st.session_state.best = None

# Load dataset
df = pd.read_csv("/Users/saathvik2005/Documents/UnifiedM/PR2/data/nassau_dataset.csv")

encoders = joblib.load("/Users/saathvik2005/Documents/UnifiedM/PR2/models/encoders.pkl")

# Sidebar
st.sidebar.header("Simulation Inputs")

product = st.sidebar.selectbox(
    "Product",
    df["Product Name"].unique()
)

region = st.sidebar.selectbox(
    "Region",
    df["Region"].unique()
)

ship_mode = st.sidebar.selectbox(
    "Ship Mode",
    df["Ship Mode"].unique()
)

units = st.sidebar.slider("Units",1,100,10)

cost = st.sidebar.number_input("Cost",1.0,1000.0,50.0)

division = df[df["Product Name"]==product]["Division"].iloc[0]

# Encode input
input_df = pd.DataFrame({
"Ship Mode":[encoders["Ship Mode"].transform([ship_mode])[0]],
"Region":[encoders["Region"].transform([region])[0]],
"Division":[encoders["Division"].transform([division])[0]],
"Product Name":[encoders["Product Name"].transform([product])[0]],
"Units":[units],
"Cost":[cost]
})

# Factory coordinates
factory_locations = pd.DataFrame({
"Factory":[
"Lot's O' Nuts",
"Wicked Choccy's",
"Sugar Shack",
"Secret Factory",
"The Other Factory"
],
"Latitude":[
32.881893,
32.076176,
48.11914,
41.446333,
35.1175
],
"Longitude":[
-111.768036,
-81.088371,
-96.18115,
-90.565487,
-89.971107
]
})

# Example customer location
customer_location = (40.7128,-74.0060)

if st.button("Run Optimization"):

    results = simulate_factories(input_df)

    results_df = pd.DataFrame(results)

    best = recommend_factory(results)

    st.session_state.results = results_df
    st.session_state.best = best

# Display results if available
if st.session_state.results is not None:
    st.subheader("Predicted Lead Time by Factory")

    st.dataframe(st.session_state.results)

    st.success(
        f"Recommended Factory: {st.session_state.best['Factory']} "
        f"Lead Time: {st.session_state.best['Predicted Lead Time']:.2f} days"
    )

    st.bar_chart(st.session_state.results.set_index("Factory"))

    # Map
    st.subheader("Logistics Map Simulation")

    m = folium.Map(location=[39,-95],zoom_start=4)

    # Customer marker
    folium.Marker(
        customer_location,
        popup="Customer Region",
        icon=folium.Icon(color="red")
    ).add_to(m)

    for i,row in factory_locations.iterrows():

        factory_point = (row["Latitude"],row["Longitude"])

        # Highlight recommended factory
        if row["Factory"]==st.session_state.best["Factory"]:
            color="green"
        else:
            color="blue"

        folium.Marker(
            factory_point,
            popup=row["Factory"],
            icon=folium.Icon(color=color,icon="industry")
        ).add_to(m)

        # Draw route
        folium.PolyLine(
            [factory_point,customer_location],
            color=color,
            weight=2
        ).add_to(m)

        # Distance
        distance = geodesic(factory_point,customer_location).km

        folium.Marker(
            [
                (factory_point[0]+customer_location[0])/2,
                (factory_point[1]+customer_location[1])/2
            ],
            popup=f"{distance:.0f} km"
        ).add_to(m)

    st_folium(m,width=900,height=500)