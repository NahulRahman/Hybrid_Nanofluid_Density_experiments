import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Nanofluid Density Hub",
    page_icon="⚗️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-title {
        font-size: 3rem !important;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.3rem !important;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem !important;
        text-align: center;
        color: #333;
        margin: 2rem 0 1rem 0;
        text-decoration: underline;
    }
    .input-label {
        font-size: 1.1rem !important;
        font-weight: bold;
        color: #444;
    }
    .result-box {
        background-color: #f0f2f6;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 20px 0;
    }
    .result-text {
        font-size: 1.5rem !important;
        color: #333;
        margin-bottom: 10px;
    }
    .result-value {
        font-size: 2.5rem !important;
        font-weight: bold;
        color: #1f77b4;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        font-size: 1rem;
        color: #666;
    }
    .stSelectbox > div > div > select {
        font-size: 1.1rem !important;
    }
    .stNumberInput > div > div > input {
        font-size: 1.1rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the dataset"""
    # Google Sheets CSV export URL
    csv_url = "https://docs.google.com/spreadsheets/d/1gdPZ1JINyv7VsKupHKVWJgHY0dEljzW_AY2DPCWK5HM/export?format=csv"
    
    try:
        df = pd.read_csv(csv_url)
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        df['Nano Particle'] = df['Nano Particle'].str.strip()
        df['Base Fluid'] = df['Base Fluid'].str.strip()
        
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def train_model(df):
    """Train XGBoost model"""
    # Create label encoders for categorical variables
    le_nano = LabelEncoder()
    le_base = LabelEncoder()
    
    # Prepare data
    df_processed = df.copy()
    df_processed['Nano Particle'] = le_nano.fit_transform(df['Nano Particle'])
    df_processed['Base Fluid'] = le_base.fit_transform(df['Base Fluid'])
    
    # Separate features and target
    X = df_processed.drop('Density (ρ)', axis=1)
    y = df_processed['Density (ρ)']
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = XGBRegressor(random_state=42, n_estimators=100)
    model.fit(X_scaled, y)
    
    return model, scaler, le_nano, le_base

def predict_density(model, scaler, le_nano, le_base, input_data):
    """Predict density based on input parameters"""
    # Encode categorical variables
    nano_encoded = le_nano.transform([input_data['nano_particle']])[0]
    base_encoded = le_base.transform([input_data['base_fluid']])[0]
    
    # Prepare feature array
    features = np.array([[
        nano_encoded,
        base_encoded,
        input_data['temperature'],
        input_data['volume_concentration'],
        input_data['density_np1'],
        input_data['density_np2'],
        input_data['density_bf'],
        input_data['volume_mix1'],
        input_data['volume_mix2']
    ]])
    
    # Scale features
    features_scaled = scaler.transform(features)
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    
    return prediction

# Main app
def main():
    # Load data and train model
    df = load_and_preprocess_data()
    
    if df is None:
        st.error("Failed to load dataset. Please check the data source.")
        return
    
    # Train model
    model, scaler, le_nano, le_base = train_model(df)
    
    # Header
    st.markdown('<h3 class="main-title">Nanofluid Density Hub</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">A Web Application predicting The density For Hybrid Nanofluids</p>', unsafe_allow_html=True)
    
    # Input section
    st.markdown('<h2 class="section-header">Give Input Parameters</h2>', unsafe_allow_html=True)
    
    # Create 3x3 grid for inputs
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<p class="input-label">Nano Particle</p>', unsafe_allow_html=True)
        nano_particle = st.selectbox(
            "",
            df['Nano Particle'].unique(),
            key="nano_particle"
        )
        
        st.markdown('<p class="input-label">Volume Concentration (φ)</p>', unsafe_allow_html=True)
        volume_concentration = st.number_input(
            "",
            min_value=0.0,
            max_value=1.0,
            value=0.05,
            step=0.01,
            key="volume_concentration"
        )
        
        st.markdown('<p class="input-label">Density of Base Fluid (ρbf)</p>', unsafe_allow_html=True)
        density_bf = st.number_input(
            "",
            min_value=500.0,
            max_value=1500.0,
            value=998.29,
            step=0.01,
            key="density_bf"
        )
    
    with col2:
        st.markdown('<p class="input-label">Base Fluid</p>', unsafe_allow_html=True)
        base_fluid = st.selectbox(
            "",
            df['Base Fluid'].unique(),
            key="base_fluid"
        )
        
        st.markdown('<p class="input-label">Density of Nano Particle 1 (ρnp)</p>', unsafe_allow_html=True)
        density_np1 = st.number_input(
            "",
            min_value=1000,
            max_value=10000,
            value=3890,
            step=1,
            key="density_np1"
        )
        
        st.markdown('<p class="input-label">Volume Mixture of Particle 1</p>', unsafe_allow_html=True)
        volume_mix1 = st.number_input(
            "",
            min_value=0,
            max_value=100,
            value=20,
            step=1,
            key="volume_mix1"
        )
    
    with col3:
        st.markdown('<p class="input-label">Temperature (°C)</p>', unsafe_allow_html=True)
        temperature = st.number_input(
            "",
            min_value=10,
            max_value=100,
            value=20,
            step=1,
            key="temperature"
        )
        
        st.markdown('<p class="input-label">Density of Nano Particle 2 (ρnp)</p>', unsafe_allow_html=True)
        density_np2 = st.number_input(
            "",
            min_value=1000,
            max_value=10000,
            value=2220,
            step=1,
            key="density_np2"
        )
        
        st.markdown('<p class="input-label">Volume Mixture of Particle 2</p>', unsafe_allow_html=True)
        volume_mix2 = st.number_input(
            "",
            min_value=0,
            max_value=100,
            value=80,
            step=1,
            key="volume_mix2"
        )
    
    # Submit button
    st.markdown("<br>", unsafe_allow_html=True)
    col_center = st.columns([1, 1, 1])
    with col_center[1]:
        submit_button = st.button("Submit to see the Density (ρ)", type="primary", use_container_width=True)
    
    # Result section
    if submit_button:
        # Validate volume mixture percentages
        if volume_mix1 + volume_mix2 != 100:
            st.error("Volume Mixture of Particle 1 and Particle 2 must sum to 100%")
        else:
            # Prepare input data
            input_data = {
                'nano_particle': nano_particle,
                'base_fluid': base_fluid,
                'temperature': temperature,
                'volume_concentration': volume_concentration,
                'density_np1': density_np1,
                'density_np2': density_np2,
                'density_bf': density_bf,
                'volume_mix1': volume_mix1,
                'volume_mix2': volume_mix2
            }
            
            # Make prediction
            try:
                predicted_density = predict_density(model, scaler, le_nano, le_base, input_data)
                
                # Display result
                st.markdown("""
                <div class="result-box">
                    <p class="result-text">The Density (ρ) of the Hybrid Nanofluid is</p>
                    <p class="result-value">{:.3f} kg/m³</p>
                </div>
                """.format(predicted_density), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    else:
        # Empty result placeholder
        st.markdown("""
        <div class="result-box">
            <p class="result-text">The Density (ρ) of the Hybrid Nanofluid is</p>
            <p class="result-value">---</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>The XGBoost Based Hybrid Nanofluid Density Prediction Web application is Developed By <strong> Md. Nahul Rahman</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()