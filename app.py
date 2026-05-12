import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. LOAD THE MODEL
# Ensure this file is in the same folder as app.py
try:
    model = joblib.load('spanish_energy_model.pkl')
except:
    st.error("Model file not found. Please ensure 'spanish_energy_model.pkl' is in the project folder.")

# 2. PAGE CONFIGURATION
st.set_page_config(page_title="Spain Energy AI", layout="wide")
st.title("⚡ Spanish Electricity Price Predictor (2015-2018)")
st.markdown("""
    This application utilizes an **Explainable Random Forest Regressor** to analyze how renewable energy 
    impacts the Spanish market clearing price. Use the sidebar to simulate different market conditions.
""")

# 3. SIDEBAR INPUTS
st.sidebar.header("🔧 Market Variables")

def get_user_inputs():
    # Primary Sliders (The ones that change the price the most)
    wind = st.sidebar.slider('Wind Generation (MW)', 0, 15000, 5500)
    solar = st.sidebar.slider('Solar Generation (MW)', 0, 6000, 1500)
    load = st.sidebar.slider('Total Grid Load (MW)', 15000, 45000, 28000)
    price_lag = st.sidebar.number_input("Yesterday's Price (€/MWh)", value=55.0)
    
    # Secondary Controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("Time Context")
    hr = st.sidebar.selectbox("Hour of Day", list(range(24)), index=12)
    day = st.sidebar.selectbox("Day of Week (0=Mon, 6=Sun)", list(range(7)), index=2)
    mo = st.sidebar.selectbox("Month (1-12)", list(range(1, 13)), index=5)

    # 4. DATA CONSOLIDATION (The Fix for the ValueError)
    # We must match the exact 13 features used in training
    data = {
        'generation solar': solar,
        'generation wind onshore': wind,
        'generation hydro water reservoir': 2500.0,  # Mean baseline
        'generation hydro run-of-river and poundage': 950.0,
        'generation biomass': 380.0,
        'generation other renewable': 90.0,
        'forecast solar day ahead': solar + 10, # Realistic forecast correlation
        'forecast wind onshore day ahead': wind + 50,
        'total load actual': load,
        'price_lag_1': price_lag,
        'hour': hr,
        'dayofweek': day,
        'month': mo
    }
    
    # Create DataFrame and ensure column order is identical to training
    # (Scikit-learn is very strict about order!)
    return pd.DataFrame(data, index=[0])

input_df = get_user_inputs()

# 5. MAIN DASHBOARD DISPLAY
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Scenario Parameters")
    st.dataframe(input_df.T.rename(columns={0: "Value"}))

with col2:
    st.subheader("🎯 Model Prediction")
    if st.button("Calculate Market Price", use_container_width=True):
        # The actual prediction step
        prediction = model.predict(input_df)[0]
        
        # Display the result with a large metric
        st.metric(label="Predicted Clearing Price", value=f"€{prediction:.2f}/MWh")
        
        # Add Economic Insight for the Conference Paper
        baseline_price = 57.93 # From our previous calculation
        diff = prediction - baseline_price
        
        if diff < 0:
            st.success(f"This scenario results in a price **€{abs(diff):.2f} lower** than the historical average.")
        else:
            st.warning(f"This scenario results in a price **€{diff:.2f} higher** than the historical average.")

# 6. CONFERENCE PAPER VISUAL (Feature Importance)
st.divider()
st.subheader("💡 Why is the price this high/low?")
st.write("The model uses **Feature Importance** to weight variables. 'Price Lag' anchors the value, while 'Wind' acts as a primary suppressor.")