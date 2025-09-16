import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Feature list (must match the model!)
important_features = [
    'hdlngth', 'skullw', 'case', 'footlgth',
    'eye', 'chest', 'earconch', 'totlngth',
    'belly', 'taill'
]

st.title("Possum Age Prediction App")
st.write("Enter possum measurements to predict age:")

# Sidebar for user input
def user_input_features():
    input_dict = {}
    for feature in important_features:
        if feature == 'case':
            input_dict[feature] = st.number_input(f"{feature}", min_value=0, max_value=50, value=1)
        else:
            input_dict[feature] = st.number_input(f"{feature}", value=0.0)
    features_df = pd.DataFrame([input_dict])
    return features_df

input_df = user_input_features()

# Load model
with open("best_possum_rf.pkl", "rb") as f:
    model = pickle.load(f)

if st.button("Predict Age"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Possum Age: {prediction:.2f} years")
