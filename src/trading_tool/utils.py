import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from scipy.stats import norm
import plotly.graph_objects as go 

def clean_data(path1, path2):
    """
    We call this function everytime after reading the 2 csv files if we want to clean our data.
    This function removes the outliers which are error in annotations (outlier gas price) and fill Nan values.
    """

    # We load the dataframes and rename Unnamed: 0 in time(hour) and time(day)
    path = "/home/lucien/Bureau/Lucien_Rondier_Work/data/"
    df1 = pd.read_csv(path + path1)
    df2 = pd.read_csv(path + path2)
    df1 = df1.rename(columns={'Unnamed: 0': 'time(hour)'})
    df2 = df2.rename(columns={'Unnamed: 0': 'time(day)'})

    # We clean df1: this is just a forward fill on 3 columns (see the previous notebook to see why)
    df1[['power.de.avail.aggr.lignite.mw.h.obs.cet.3mv', 
    'power.de.avail.aggr.nuclear.mw.h.obs.cet.3mv', 
    'power.de.avail.aggr.coal.mw.h.obs.cet.3mv']] = df1[['power.de.avail.aggr.lignite.mw.h.obs.cet.3mv', 
                                                        'power.de.avail.aggr.nuclear.mw.h.obs.cet.3mv', 
                                                        'power.de.avail.aggr.coal.mw.h.obs.cet.3mv']].ffill()

    # We clean df2: we delete an outlier and make forward fill and then one backward fill
    target_date = '2023-04-11' # target 
    target_index = df2[df2['time(day)'] == target_date].index[0]
    df2.loc[target_index, 'price.naturalgas.ncg.delivery.powernext.eur.mwh'] = df2.loc[target_index - 1, 'price.naturalgas.ncg.delivery.powernext.eur.mwh']

    df2[['power.price.co2.eua.eurot.d.obs', 'coal.price.api2_ara.euromwh.d.obs']] = df2[
        ['power.price.co2.eua.eurot.d.obs', 'coal.price.api2_ara.euromwh.d.obs']
    ].ffill()

    df2[['power.price.co2.eua.eurot.d.obs', 'coal.price.api2_ara.euromwh.d.obs']] = df2[
        ['power.price.co2.eua.eurot.d.obs', 'coal.price.api2_ara.euromwh.d.obs']
    ].bfill()

    # Now we concatenate the two dataframes, repeating the 3 same prices of the day for every hour thanks to a left join

    df1['time(day)'] = pd.to_datetime(df1['time(hour)']).dt.date

    # Conversion de la colonne 'time(day)' de df2 au format date pour correspondre à df1
    df2['time(day)'] = pd.to_datetime(df2['time(day)']).dt.date

    # join df1 and df2
    df_merged = pd.merge(df1, df2, on='time(day)', how='left')

    # we delete the day time column
    df_merged = df_merged.drop(columns=['time(day)'])

    return df_merged

# Function to detect anormal value of a feature to make n alert to the trader

import pandas as pd
import numpy as np
from scipy.stats import zscore
import streamlit as st

def detect_anomalies_streamlit(df, date_str, z_threshold=2.0):
    # Rename columns for better readability
    column_renaming = {
        'power.prod.wind.de.mw.h.fcst.ecens.p50': 'Wind Forecast P50',
        'power.prod.solar.de.mw.h.fcst.ecens.p50': 'Solar Forecast P50',
        'power.de.avail.aggr.lignite.mw.h.obs.cet.3mv': 'Availability Lignite',
        'power.de.avail.aggr.nuclear.mw.h.obs.cet.3mv': 'Availability Nuclear',
        'power.de.avail.aggr.coal.mw.h.obs.cet.3mv': 'Availability Coal',
        'price.naturalgas.ncg.delivery.powernext.eur.mwh': 'price_naturalgas',
        'power.price.co2.eua.eurot.d.obs': 'price_co2',
        'coal.price.api2_ara.euromwh.d.obs': 'price_coal'
    }
    df = df.rename(columns=column_renaming)
    
    # Ensure 'time(hour)' is in datetime format
    df['time(hour)'] = pd.to_datetime(df['time(hour)'])
    
    # Calculate Z-scores for features
    feature_columns = df.columns.difference(['time(hour)'])
    df_z = df[feature_columns].apply(zscore)

    # Find the row for the specified date
    target_row = df[df['time(hour)'] == pd.to_datetime(date_str)]
    
    if target_row.empty:
        st.error(f"No data available for date {date_str}.")
        return

    # Get the Z-scores for the specified date
    target_z_scores = df_z.loc[target_row.index[0]]
    
    # Identify high and low anomalies
    high_anomalies = target_z_scores[target_z_scores > z_threshold].index.tolist()
    low_anomalies = target_z_scores[target_z_scores < -z_threshold].index.tolist()

    # Display results with styling
    st.title(f"Anomaly Detection for {date_str}")

    # High Anomalies
    if high_anomalies:
        st.markdown("<h4 style='color:green;'>Anomalously High Features:</h4>", unsafe_allow_html=True)
        for feature in high_anomalies:
            st.markdown(f"<span style='color:green; font-size:20px;'>{feature}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<h4>No features are anomalously high.</h4>", unsafe_allow_html=True)

    # Low Anomalies
    if low_anomalies:
        st.markdown("<h4 style='color:red;'>Anomalously Low Features:</h4>", unsafe_allow_html=True)
        for feature in low_anomalies:
            st.markdown(f"<span style='color:red; font-size:20px;'>{feature}</span>", unsafe_allow_html=True)
    else:
        st.markdown("<h4>No features are anomalously low.</h4>", unsafe_allow_html=True)


# Function that display in streamlit the return of the day for price.naturalgas.ncg.delivery.powernext.eur.mwh,power.price.co2.eua.eurot.d.obs,coal.price.api2_ara.euromwh.d.obs

def display_daily_returns(df, date_str):
    # Column renaming for clarity
    column_renaming = {
        'power.prod.wind.de.mw.h.fcst.ecens.p50': 'Wind Forecast P50',
        'power.prod.solar.de.mw.h.fcst.ecens.p50': 'Solar Forecast P50',
        'power.de.avail.aggr.lignite.mw.h.obs.cet.3mv': 'Availability Lignite',
        'power.de.avail.aggr.nuclear.mw.h.obs.cet.3mv': 'Availability Nuclear',
        'power.de.avail.aggr.coal.mw.h.obs.cet.3mv': 'Availability Coal'
    }
    df = df.rename(columns=column_renaming)

    # Convert `date_str` to a datetime object
    target_date = pd.to_datetime(date_str)
    start_date = target_date - pd.Timedelta(hours=1)
    
    # Calculate daily returns as the percentage change for each renamed column
    columns = list(column_renaming.values())
    daily_returns = df[columns].pct_change() * 100

    # Ensure 'time(hour)' is in datetime format for filtering
    df['time(hour)'] = pd.to_datetime(df['time(hour)'])
    
    # Filter daily returns DataFrame for the specified date range
    filtered_returns = daily_returns[(df['time(hour)'] >= start_date) & (df['time(hour)'] <= target_date)]

    # Display daily returns for each column with arrows and colors in Streamlit
    st.title(f"Daily Returns from {start_date} to {target_date} for Selected Commodities")
    for column in columns:
        # Get the daily return value for the target date
        return_value = filtered_returns.loc[df['time(hour)'] == target_date, column].values[0]
        
        # Determine arrow direction and color based on return
        if return_value > 0:
            arrow = "↑"
            color = "green"
        else:
            arrow = "↓"
            color = "red"
        
        # Display the return with arrow and color
        st.markdown(f"### {column}")
        st.markdown(f"<span style='color:{color}; font-size:24px;'>{arrow} {abs(return_value):.2f}%</span>", unsafe_allow_html=True)


def create_train_display(df, input_date):
    # Ensure 'time(hour)' is in datetime format
    df['time(hour)'] = pd.to_datetime(df['time(hour)'])

    # Number of hours we look back
    lookback = 5

    # Prepare data for multivariate time series prediction
    X, y = [], []
    for i in range(len(df) - lookback):
        X.append(df.iloc[i:i+lookback].drop(columns=['time(hour)']).values)
        y.append(df.iloc[i + lookback]['power.de.price.day_ahead.euromwh.h.obs.cet.epex'])  # Target value at the following hour

    # Convert to numpy arrays
    X, y = np.array(X), np.array(y)

    # Flatten X for XGBRegressor (requires 2D input)
    n_samples, lookback, n_features = X.shape
    X_flat = X.reshape(n_samples, lookback * n_features)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_flat, y, test_size=0.2, shuffle=False)

    # Initialize, train, and predict with XGBRegressor
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    model.fit(X_train, y_train)

    # Predict on the test set to calculate residuals
    y_pred_test = model.predict(X_test)
    residuals = y_test - y_pred_test
    std_residuals = np.std(residuals)

    # Find the index corresponding to input_date
    input_date = pd.to_datetime(input_date)
    input_index = df[df['time(hour)'] == input_date].index[0]

    # Prepare the input features for the next hour after `input_date`
    X_next = df.iloc[input_index - lookback + 1:input_index + 1].drop(columns=['time(hour)']).values.flatten().reshape(1, -1)
    y_next_pred = model.predict(X_next)[0]

    # Calculate confidence interval for the prediction
    alpha = 0.1  # 90% confidence level
    z = norm.ppf(1 - alpha / 2)
    lower_bound = y_next_pred - z * std_residuals
    upper_bound = y_next_pred + z * std_residuals

    # Plot the prediction with confidence interval
    st.subheader("Prediction for the Next Hour after the Specified Date with Confidence Interval")
    st.write(f"Prediction for the next hour after {input_date}:")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
    x=df['time(hour)'].iloc[input_index - 19:input_index + 1], 
    y=y[input_index - lookback - 19:input_index - lookback + 1], 
    mode='lines', 
    name='Actual Price'
    ))
    fig.add_trace(go.Scatter(x=[input_date + pd.Timedelta(hours=1)],
                             y=[y_next_pred], mode='markers', name='Prediction',
                             marker=dict(color='red', size=10)))
    fig.add_trace(go.Scatter(x=[input_date + pd.Timedelta(hours=1)],
                             y=[lower_bound], mode='markers', name='Lower Bound',
                             marker=dict(color='blue', size=7, symbol='arrow-bar-down')))
    fig.add_trace(go.Scatter(x=[input_date + pd.Timedelta(hours=1)],
                             y=[upper_bound], mode='markers', name='Upper Bound',
                             marker=dict(color='blue', size=7, symbol='arrow-bar-up')))

    # Customize layout
    fig.update_layout(title="Prediction with 95% Confidence Interval",
                      xaxis_title="Time (hour)",
                      yaxis_title="Price (EUR/MWh)",
                      showlegend=True)

    st.plotly_chart(fig)

    # Display the predicted value and confidence interval
    st.write(f"Predicted value: {y_next_pred:.2f} EUR/MWh")
    st.write(f"95% Confidence Interval: [{lower_bound:.2f}, {upper_bound:.2f}] EUR/MWh")

