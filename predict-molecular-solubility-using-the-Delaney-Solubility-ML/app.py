import streamlit as st
import pandas as pd
import joblib
import numpy as np

# App title and description
st.title("Molecular Solubility Predictor")
st.write("Enter molecular features to predict aqueous solubility (logS) using Linear Regression or Random Forest models trained on the Delaney Dataset.")

# Load models
@st.cache_resource  # Cache models for faster loading
def load_models():
    lr_model = joblib.load('lr_model.pkl')
    rf_model = joblib.load('rf_model.pkl')
    return lr_model, rf_model

lr_model, rf_model = load_models()

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Choose Model:", ["Linear Regression", "Random Forest"])

# Input fields for features
st.sidebar.header("Input Features")
mol_logp = st.sidebar.number_input("MolLogP (Hydrophobicity)", value=2.0, step=0.1, format="%.2f")
mol_wt = st.sidebar.number_input("MolWt (Molecular Weight)", value=200.0, step=1.0, format="%.1f")
num_rot_bonds = st.sidebar.number_input("NumRotatableBonds (Flexibility)", value=2, step=1)
arom_prop = st.sidebar.number_input("AromaticProportion", value=0.5, step=0.1, format="%.2f")

# Prepare input data
if st.sidebar.button("Predict Solubility"):
    input_data = pd.DataFrame({
        'MolLogP': [mol_logp],
        'MolWt': [mol_wt],
        'NumRotatableBonds': [num_rot_bonds],
        'AromaticProportion': [arom_prop]
    })
    
    # Predict
    if model_choice == "Linear Regression":
        prediction = lr_model.predict(input_data)[0]
        model_name = "Linear Regression"
    else:
        prediction = rf_model.predict(input_data)[0]
        model_name = "Random Forest"
    
    # Display result
    st.header(f"Predicted logS (using {model_name}): {prediction:.4f}")
    st.write("This is the logarithm of aqueous solubility in mol/L.")

# Optional: Add a section to show model performance (hardcoded from training)
st.sidebar.markdown("---")
st.sidebar.write("Model Performance (on test set):")
st.sidebar.write("**Linear Regression:** MSE ≈ 1.05, R² ≈ 0.68")
st.sidebar.write("**Random Forest:** MSE ≈ 0.85, R² ≈ 0.75")
# Update these values based on your actual training output

# Optional: Visualization (e.g., load a plot from notebook if saved as image)
# st.image("scatter_plot.png")  # If you save a predicted vs actual plot