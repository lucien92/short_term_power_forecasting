import pandas as pd
import streamlit as st
from datetime import datetime
from utils import display_daily_returns, clean_data, detect_anomalies_streamlit, create_train_display

# Streamlit app layout
st.title("Short term power trend")

# Input section for date and time
date_input = st.text_input("Enter the date and time (format: YYYY-MM-DD HH:MM:SS+00:00)", value="2022-11-06 13:00:00+00:00")
z_threshold = st.slider("Choose the Z-score Threshold", 0.0, 5.0, 2.0, step=0.1)

# Button to display daily returns
if st.button("Display Daily Returns"):

    col1, col2 = st.columns(2)

    # Load and clean data
    path1, path2 = "data1.csv", "data2.csv"
    df = clean_data(path1, path2)
    
    # Display daily returns in the left column
    with col1:
        st.subheader("Daily Returns")
        display_daily_returns(df, date_input)
    
    # Display anomaly detection in the right column
    with col2:
        st.subheader("Anomaly Detection")
        detect_anomalies_streamlit(df, date_input, z_threshold)

    # display in the middle

    create_train_display(df, date_input)
