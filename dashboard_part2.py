import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np # Import numpy for linear regression

# Set page configuration
st.set_page_config(layout="wide", page_title="Environmental Data Analysis Dashboard")

# --- Generic Data Loading and Preprocessing ---
@st.cache_data
def _load_and_preprocess_data(file_path, date_cols, rename_cols_map, numeric_cols):
    """
    Generic function to load and preprocess data from a CSV.
    Combines year/month into a 'Date' column, renames specified columns,
    and selects relevant numeric and date/location columns.
    """
    try:
        df = pd.read_csv(file_path)
        # Combine year and month to create a Date column
        df['Date'] = pd.to_datetime(df[date_cols[0]].astype(str) + '-' + df[date_cols[1]].astype(str) + '-01')
        
        # Rename columns if specified
        if rename_cols_map:
            df.rename(columns=rename_cols_map, inplace=True)
            
        # Select relevant columns, ensuring they exist
        selected_cols = ['Date', 'Location'] + [col for col in numeric_cols if col in df.columns]
        df = df[selected_cols]
        
        return df
    except FileNotFoundError:
        st.error(f"Error: '{file_path}' not found. Please ensure the file is in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing for '{file_path}': {e}")
        return pd.DataFrame()

# Load data using the generic function
water_df = _load_and_preprocess_data(
    'water_quality_filled (1).csv',
    ['YEAR', 'MONTH'],
    {'LOCATION': 'Location'},
    ['SURFACE_WATER_TEMP', 'MIDDLE_WATER_TEMP', 'BOTTOM_WATER_TEMP',
     'PH_LEVEL', 'AMMONIA', 'NITRATE_NITRITE', 'PHOSPHATE', 'DISSOLVED_OXYGEN']
)

meteorological_df = _load_and_preprocess_data(
    'completed_meteorological_filled.csv',
    ['YEAR', 'MONTH'],
    {'LOCATION': 'Location'},
    ['RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND SPEED', 'WIND DIRECTION']
)

volcanic_df = _load_and_preprocess_data(
    'completed_volcanic_flux_filled.csv',
    ['YEAR', 'MONTH'],
    {'LOCATION': 'Location'},
    ['CO2 FLUX', 'SO2 FLUX']
)


# --- Function to display the Introduction Section ---
def display_introduction():
    """Displays a short introduction to the dashboard."""
    st.markdown("""
    ## Welcome to the Environmental Data Analysis Dashboard!

    Explore vital environmental data including water quality, meteorological conditions, and volcanic activity. This dashboard provides interactive visualizations and insights to help understand trends and patterns in environmental monitoring. For predictions, this dashboard utilizes concepts from advanced time-series forecasting, including those inspired by **Hybrid CNN-LSTM** architectures for potential future implementations.
    """)

# --- Function to calculate a simplified Composite Water Quality Index (WQI) for a row ---
def calculate_composite_wqi(row):
    """
    Calculates a simplified composite Water Quality Index (WQI) for a single row of water quality data.
    This is an illustrative calculation and not based on official, complex WQI standards.
    It assigns a score (0-100) to each relevant parameter and then calculates a weighted average.
    """
    scores = {}
    
    # Define quality rating for each parameter (simplified examples)
    # These should be based on regulatory standards for actual WQI
    # Each parameter score is based on its value relative to ideal/acceptable ranges.

    # pH Score: Ideal 6.5-8.5
    if 'PH_LEVEL' in row and pd.notna(row['PH_LEVEL']):
        ph = row['PH_LEVEL']
        if 6.5 <= ph <= 8.5: scores['PH_LEVEL'] = 100
        elif (6.0 <= ph < 6.5) or (8.5 < ph <= 9.0): scores['PH_LEVEL'] = 75
        elif (5.5 <= ph < 6.0) or (9.0 < ph <= 9.5): scores['PH_LEVEL'] = 50
        else: scores['PH_LEVEL'] = 25
    else: scores['PH_LEVEL'] = 0 # Assume 0 score if data is missing

    # Dissolved Oxygen Score: Higher is generally better for aquatic life, typically > 5 mg/L good
    if 'DISSOLVED_OXYGEN' in row and pd.notna(row['DISSOLVED_OXYGEN']):
        do = row['DISSOLVED_OXYGEN']
        if do >= 7.0: scores['DISSOLVED_OXYGEN'] = 100
        elif 5.0 <= do < 7.0: scores['DISSOLVED_OXYGEN'] = 75
        elif 3.0 <= do < 5.0: scores['DISSOLVED_OXYGEN'] = 50
        else: scores['DISSOLVED_OXYGEN'] = 25
    else: scores['DISSOLVED_OXYGEN'] = 0

    # Ammonia Score: Lower is better (toxicity)
    if 'AMMONIA' in row and pd.notna(row['AMMONIA']):
        ammonia = row['AMMONIA']
        if ammonia <= 0.05: scores['AMMONIA'] = 100
        elif 0.05 < ammonia <= 0.5: scores['AMMONIA'] = 75
        elif 0.5 < ammonia <= 2.0: scores['AMMONIA'] = 50
        else: scores['AMMONIA'] = 25
    else: scores['AMMONIA'] = 0

    # Nitrate-Nitrite Score: Lower is better (eutrophication, toxicity)
    if 'NITRATE_NITRITE' in row and pd.notna(row['NITRATE_NITRITE']):
        nitrate = row['NITRATE_NITRITE']
        if nitrate <= 1.0: scores['NITRATE_NITRITE'] = 100
        elif 1.0 < nitrate <= 10.0: scores['NITRATE_NITRITE'] = 75
        elif 10.0 < nitrate <= 50.0: scores['NITRATE_NITRITE'] = 50
        else: scores['NITRATE_NITRITE'] = 25
    else: scores['NITRATE_NITRITE'] = 0

    # Phosphate Score: Lower is better (eutrophication)
    if 'PHOSPHATE' in row and pd.notna(row['PHOSPHATE']):
        phosphate = row['PHOSPHATE']
        if phosphate <= 0.1: scores['PHOSPHATE'] = 100
        elif 0.1 < phosphate <= 0.5: scores['PHOSPHATE'] = 75
        elif 0.5 < phosphate <= 1.0: scores['PHOSPHATE'] = 50
        else: scores['PHOSPHATE'] = 25
    else: scores['PHOSPHATE'] = 0

    # Temperature Score (for Surface, Middle, Bottom Water Temp): Closer to optimal is better
    # Assuming optimal range 20-25 C, adjust as per specific aquatic ecosystem needs
    temp_params = ['SURFACE_WATER_TEMP', 'MIDDLE_WATER_TEMP', 'BOTTOM_WATER_TEMP']
    for temp_param in temp_params:
        if temp_param in row and pd.notna(row[temp_param]):
            temp = row[temp_param]
            if 20 <= temp <= 25: scores[temp_param] = 100
            elif (15 <= temp < 20) or (25 < temp <= 30): scores[temp_param] = 75
            elif (10 <= temp < 15) or (30 < temp <= 35): scores[temp_param] = 50
            else: scores[temp_param] = 25
        else: scores[temp_param] = 0

    # Define weights for each parameter. Sum of weights should ideally be 1.0 or normalized.
    # These are illustrative weights. Proper weights need expert input.
    weights = {
        'PH_LEVEL': 0.15,
        'DISSOLVED_OXYGEN': 0.25,
        'AMMONIA': 0.20,
        'NITRATE_NITRITE': 0.15,
        'PHOSPHATE': 0.10,
        'SURFACE_WATER_TEMP': 0.05,
        'MIDDLE_WATER_TEMP': 0.05,
        'BOTTOM_WATER_TEMP': 0.05,
    }

    composite_wqi = 0
    total_effective_weight = 0

    for param, weight in weights.items():
        if param in scores and scores[param] is not None: # Ensure score exists
            composite_wqi += scores[param] * weight
            total_effective_weight += weight

    if total_effective_weight == 0:
        return "N/A" # No valid parameters to calculate WQI

    return f"{composite_wqi / total_effective_weight:.2f}"


# --- Function to display Water Quality Analysis Section ---
def display_water_quality_analysis(df):
    st.header("Water Quality Analysis")
    st.markdown("Provides insights into water quality parameters, trends, and historical data.")

    if df.empty:
        st.error("Cannot proceed with Water Quality Analysis as no data was loaded.")
        return

    # --- Sidebar Filters for Water Quality ---
    st.sidebar.header("Filter Water Quality Data")
    selected_water_location = st.sidebar.multiselect(
        "Select Water Location(s):",
        options=df['Location'].unique(),
        default=df['Location'].unique(),
        key='water_location_filter'
    )
    water_filtered_df = df[df['Location'].isin(selected_water_location)]

    if not water_filtered_df.empty:
        water_min_date = water_filtered_df['Date'].min().to_pydatetime()
        water_max_date = water_filtered_df['Date'].max().to_pydatetime()
        water_date_range = st.sidebar.slider(
            "Select Water Date Range:",
            value=(water_min_date, water_max_date),
            format="YYYY-MM-DD",
            key='water_date_range_filter'
        )
        water_filtered_df = water_filtered_df[(water_filtered_df['Date'] >= pd.to_datetime(water_date_range[0])) &
                                              (water_filtered_df['Date'] <= pd.to_datetime(water_date_range[1]))]
    else:
        st.warning("No water quality data available for the selected locations to determine date range.")
        water_filtered_df = pd.DataFrame()


    # --- Water Quality Parameter Trends ---
    st.subheader("Parameter Trends")
    water_parameter_options = [col for col in df.columns if col not in ['Date', 'Location']]
    if not water_parameter_options:
        st.warning("No measurable water quality parameters found in the loaded data.")
        selected_water_parameter = None
    else:
        selected_water_parameter = st.selectbox("Choose a water quality parameter to analyze:", water_parameter_options, key='select_water_param')

    if selected_water_parameter and not water_filtered_df.empty:
        st.write(f'Trend of {selected_water_parameter} Over Time by Location')
        try:
            line_chart_data = water_filtered_df.pivot_table(index='Date', columns='Location', values=selected_water_parameter, aggfunc='mean').fillna(0)
            st.line_chart(line_chart_data)

        except Exception as e:
            st.warning(f"Could not render water quality line chart. Please adjust filters or check data. Error: {e}")
            st.dataframe(water_filtered_df[['Date', 'Location', selected_water_parameter]])
    elif selected_water_parameter:
        st.info("No water quality data available for the selected filters for plotting trends.")
    else:
        if not df.empty:
            st.info("Please select a water quality parameter to view trends.")

    st.markdown("---")

    # --- Water Quality Summary Statistics ---
    st.subheader("Summary Statistics")
    if not water_filtered_df.empty and water_parameter_options:
        st.dataframe(water_filtered_df.groupby('Location')[water_parameter_options].describe().T.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}), use_container_width=True)
    else:
        st.info("Apply filters and select a water quality parameter to see summary statistics.")

    st.markdown("---")

    # --- Water Quality Parameter Distribution (Bar Chart of Averages) ---
    st.subheader("Parameter Distribution (Averages)")
    if selected_water_parameter and not water_filtered_df.empty:
        st.write(f'Average {selected_water_parameter} by Location')
        avg_data = water_filtered_df.groupby('Location')[selected_water_parameter].mean().reset_index()
        st.bar_chart(avg_data.set_index('Location'))
    elif selected_water_parameter:
        st.info("No water quality data available for the selected filters for plotting distribution.")
    else:
        if not df.empty:
            st.info("Please select a water quality parameter to view distribution.")

    st.markdown("---")

    # --- Water Quality Historical Data Section ---
    st.subheader("Historical Data")
    if not water_filtered_df.empty:
        # Calculate and add WQI to the historical data DataFrame
        # Ensure 'WQI (Composite)' is a string type to handle "N/A"
        water_filtered_df['WQI (Composite)'] = water_filtered_df.apply(calculate_composite_wqi, axis=1)
        st.write("Below is a table displaying the historical water quality data and Composite WQI based on your selected filters.")
        st.dataframe(water_filtered_df, use_container_width=True)
    else:
        st.info("No historical water quality data available for the current filters. Please adjust your selections in the sidebar.")

    st.markdown("---")

    # --- Water Quality Index (WQI) Trend ---
    st.subheader("Water Quality Index (WQI) Trend")
    if not water_filtered_df.empty:
        # Convert WQI to numeric for plotting, coercing 'N/A' to NaN
        wqi_plot_data = water_filtered_df.copy()
        wqi_plot_data['WQI (Composite)'] = pd.to_numeric(wqi_plot_data['WQI (Composite)'], errors='coerce')

        if not wqi_plot_data['WQI (Composite)'].dropna().empty:
            # Pivot table for WQI trend by location
            wqi_line_chart_data = wqi_plot_data.pivot_table(index='Date', columns='Location', values='WQI (Composite)', aggfunc='mean').fillna(0)
            st.line_chart(wqi_line_chart_data)
            st.info("This chart displays the Composite Water Quality Index (WQI) trend over time for selected locations. Higher values indicate better water quality.")
        else:
            st.warning("Not enough valid WQI data points to display a trend. Ensure relevant parameters are present and not all are 'N/A'.")
    else:
        st.info("No water quality data available to compute WQI trend. Please adjust your selections.")


# --- Function to display Historical Pollutant Levels Analysis Section ---
def display_pollutant_levels_analysis(water_df_full, meteorological_df_full, volcanic_df_full):
    st.header("Historical Pollutant Levels Analysis")
    st.markdown("Provides detailed trends for key water pollutants and allows overlaying related environmental data.")

    # --- Removed Sidebar Filters for Pollutant Data ---
    st.sidebar.header("Display Options for Pollutant Data")

    # Use buttons for display options
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button('Water Quality Parameters', key='wq_params_btn'):
            st.session_state['selected_pollutant_display'] = 'Water Quality Parameters'
    with col2:
        if st.button('WQ + Meteorological', key='wq_met_params_btn'):
            st.session_state['selected_pollutant_display'] = 'Water Quality Parameters + Meteorological Parameters'
    
    col3, col4 = st.sidebar.columns(2)
    with col3:
        if st.button('WQ + Volcanic Activity', key='wq_volc_params_btn'):
            st.session_state['selected_pollutant_display'] = 'Water Quality Parameters + Volcanic Activity Parameters'
    with col4:
        if st.button('WQ + Met + Volcanic', key='wq_met_volc_params_btn'):
            st.session_state['selected_pollutant_display'] = 'Water Quality Parameters + Meteorological Parameters + Volcanic Activity Parameters'

    display_option = st.session_state['selected_pollutant_display']

    # --- Display blank content as requested by the user ---
    st.info(f"Content for '{display_option}' is currently blank as per user request.")


# --- Helper Function for Prediction and Plotting ---
def _predict_and_plot_trend(df_for_prediction, param_name, prediction_datetime, latest_date_available, trend_window_months, title_prefix=""):
    """
    Predicts a future value for a given parameter based on a linear trend
    and plots the historical data along with the predicted point.
    """
    
    # Prepare historical data for plotting (average across locations for each date)
    # Ensure param_name exists in df_for_prediction columns
    if param_name not in df_for_prediction.columns:
        st.warning(f"Parameter '{param_name}' not found in the provided data. Cannot provide prediction or plot.")
        return

    historical_plot_data = df_for_prediction.groupby('Date')[param_name].mean().reset_index()
    
    # Filter for trend data based on the trend window
    start_trend_date = latest_date_available - pd.DateOffset(months=trend_window_months - 1)
    trend_data = df_for_prediction[df_for_prediction['Date'] >= start_trend_date].copy()
    trend_data = trend_data.dropna(subset=[param_name]) # Drop NaNs for this specific parameter for trend calculation
    
    predicted_value = None

    if len(trend_data) >= 2: # Need at least 2 points to calculate a linear trend
        trend_data['Date_Ordinal'] = trend_data['Date'].apply(lambda x: x.toordinal())
        X = trend_data['Date_Ordinal'].values.reshape(-1, 1)
        y = trend_data[param_name].values

        try:
            m, c = np.polyfit(X.flatten(), y, 1)
            future_date_ordinal = prediction_datetime.toordinal()
            predicted_value = m * future_date_ordinal + c

            # Apply specific clamping based on parameter
            if param_name == 'PH_LEVEL':
                predicted_value = max(0.0, min(14.0, predicted_value))
            elif param_name == 'WQI':
                predicted_value = max(0, min(100, predicted_value))
            else: # For other concentrations like Ammonia, Nitrate, Phosphate (should be non-negative)
                predicted_value = max(0.0, predicted_value)

        except np.linalg.LinAlgError:
            predicted_value = df_for_prediction[param_name].mean() # Fallback to average if trend fails
        except Exception as e:
            predicted_value = df_for_prediction[param_name].mean() # Fallback to average on other errors
            st.error(f"An error occurred during trend calculation for {param_name}: {e}")
    else:
        predicted_value = df_for_prediction[param_name].mean() # Fallback to average if not enough trend data

    if predicted_value is not None:
        predicted_point_df = pd.DataFrame({
            'Date': [prediction_datetime],
            param_name: [predicted_value]
        })
        # Use the historical_plot_data which is already averaged by Date
        combined_plot_data = pd.concat([historical_plot_data, predicted_point_df], ignore_index=True)
        combined_plot_data = combined_plot_data.sort_values(by='Date')

        st.subheader(f"{title_prefix}{param_name.replace('_', ' ').title()} Trend and Prediction")
        st.line_chart(combined_plot_data.set_index('Date'))
        st.info(f"Predicted {param_name.replace('_', ' ').title()} for {prediction_datetime.strftime('%Y-%m-%d')}: **{predicted_value:.2f}**")
        st.markdown("""
        <small><i>Note: Prediction is a simplified estimation based on a linear trend from historical data.</i></small>
        """, unsafe_allow_html=True)
    st.markdown("---")


# --- Function to display WQI Prediction Tool Section ---
def display_wqi_prediction_tool(water_df_full):
    st.header("Water Quality Index (WQI) Prediction Tool")
    st.markdown("Predict the Water Quality Index and key pollutant levels for a future date. This tool attempts to project basic trends based on historical water quality data.")

    if water_df_full.empty:
        st.warning("No water quality data loaded. Cannot provide WQI or pollutant predictions.")
        return

    # Calculate WQI for all historical data to get a baseline
    wqi_data_all = water_df_full.copy()
    wqi_data_all['WQI'] = wqi_data_all.apply(calculate_composite_wqi, axis=1)
    wqi_data_all['WQI'] = pd.to_numeric(wqi_data_all['WQI'], errors='coerce')
    
    # Filter out rows where WQI is NaN (if calculate_composite_wqi returned "N/A" for example)
    wqi_data_valid = wqi_data_all.dropna(subset=['WQI'])

    if wqi_data_valid.empty:
        st.error("No valid WQI data points found in historical data for prediction.")
        return

    # Sort data by date to ensure correct trend calculation
    wqi_data_valid = wqi_data_valid.sort_values(by='Date')

    # Get the latest date from the valid WQI data
    latest_date_available = wqi_data_valid['Date'].max()

    # Set default date for prediction to one month after the latest available data
    default_prediction_date = latest_date_available + pd.DateOffset(months=1)

    # Date input for future prediction
    prediction_date = st.date_input(
        "Select a future date for prediction:",
        value=default_prediction_date,
        min_value=latest_date_available, # User cannot select dates before the latest historical data
        key='wqi_prediction_date'
    )

    # Convert selected date to datetime object
    prediction_datetime = pd.to_datetime(prediction_date)

    if st.button("Predict"): # Changed button text to be more general
        st.subheader("Prediction Results")

        # Define trend window for consistency across all predictions
        trend_window_months = 12 
        
        # --- WQI Prediction ---
        _predict_and_plot_trend(wqi_data_valid, 'WQI', prediction_datetime, latest_date_available, trend_window_months, "Water Quality Index (")

        # --- Individual Pollutant Predictions ---
        pollutants_to_predict = ['PH_LEVEL', 'AMMONIA', 'NITRATE_NITRITE', 'PHOSPHATE']
        
        for pollutant in pollutants_to_predict:
            if pollutant in water_df_full.columns:
                _predict_and_plot_trend(water_df_full, pollutant, prediction_datetime, latest_date_available, trend_window_months)
            else:
                st.warning(f"Pollutant '{pollutant}' not found in the water quality data. Cannot provide prediction.")


    else:
        st.info("Select a future date and click 'Predict' to see estimated WQI and pollutant values. These predictions are based on historical trends found in the water quality data.")


# --- Function to display Meteorological Analysis Section ---
def display_meteorological_analysis(df):
    st.header("Meteorological Analysis")
    st.markdown("Explores weather conditions including rainfall, temperature, humidity, and wind.")

    if df.empty:
        st.error("Cannot proceed with Meteorological Analysis as no data was loaded.")
        return

    # --- Sidebar Filters for Meteorological Data ---
    st.sidebar.header("Filter Meteorological Data")
    selected_meteorological_location = st.sidebar.multiselect(
        "Select Meteorological Location(s):",
        options=df['Location'].unique(),
        default=df['Location'].unique(),
        key='meteorological_location_filter'
    )
    meteorological_filtered_df = df[df['Location'].isin(selected_meteorological_location)]

    if not meteorological_filtered_df.empty:
        meteorological_min_date = meteorological_filtered_df['Date'].min().to_pydatetime()
        meteorological_max_date = meteorological_filtered_df['Date'].max().to_pydatetime()
        meteorological_date_range = st.sidebar.slider(
            "Select Meteorological Date Range:",
            value=(meteorological_min_date, meteorological_max_date),
            format="YYYY-MM-DD",
            key='meteorological_date_range_filter'
        )
        meteorological_filtered_df = meteorological_filtered_df[(meteorological_filtered_df['Date'] >= pd.to_datetime(meteorological_date_range[0])) &
                                                              (meteorological_filtered_df['Date'] <= pd.to_datetime(meteorological_date_range[1]))]
    else:
        st.warning("No meteorological data available for the selected locations to determine date range.")
        meteorological_filtered_df = pd.DataFrame()


    # --- Meteorological Parameter Trends ---
    st.subheader("Parameter Trends")
    meteorological_parameter_options = [col for col in df.columns if col not in ['Date', 'Location']]
    if not meteorological_parameter_options:
        st.warning("No measurable meteorological parameters found in the loaded data.")
        selected_meteorological_parameter = None
    else:
        selected_meteorological_parameter = st.selectbox("Choose a meteorological parameter to analyze:", meteorological_parameter_options, key='select_meteorological_param')

    if selected_meteorological_parameter and not meteorological_filtered_df.empty:
        st.write(f'Trend of {selected_meteorological_parameter} Over Time by Location')
        try:
            line_chart_data = meteorological_filtered_df.pivot_table(index='Date', columns='Location', values=selected_meteorological_parameter, aggfunc='mean').fillna(0)
            st.line_chart(line_chart_data)
        except Exception as e:
            st.warning(f"Could not render meteorological line chart. Please adjust filters or check data. Error: {e}")
            st.dataframe(meteorological_filtered_df[['Date', 'Location', selected_meteorological_parameter]])
    elif selected_meteorological_parameter:
        st.info("No meteorological data available for the selected filters for plotting trends.")
    else:
        if not df.empty:
            st.info("Please select a meteorological parameter to view trends.")

    st.markdown("---")

    # --- Meteorological Summary Statistics ---
    st.subheader("Summary Statistics")
    if not meteorological_filtered_df.empty and meteorological_parameter_options:
        st.dataframe(meteorological_filtered_df.groupby('Location')[meteorological_parameter_options].describe().T.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}), use_container_width=True)
    else:
        st.info("Apply filters and select a meteorological parameter to see summary statistics.")

    st.markdown("---")

    # --- Meteorological Parameter Distribution (Bar Chart of Averages) ---
    st.subheader("Parameter Distribution (Averages)")
    if selected_meteorological_parameter and not meteorological_filtered_df.empty:
        st.write(f'Average {selected_meteorological_parameter} by Location')
        avg_data = meteorological_filtered_df.groupby('Location')[selected_meteorological_parameter].mean().reset_index()
        st.bar_chart(avg_data.set_index('Location'))
    elif selected_meteorological_parameter:
        st.info("No meteorological data available for the selected filters for plotting distribution.")
    else:
        if not df.empty:
            st.info("Please select a meteorological parameter to view distribution.")

    st.markdown("---")

    # --- Meteorological Historical Data Section ---
    st.subheader("Historical Data")
    if not meteorological_filtered_df.empty:
        st.write("Below is a table displaying the historical meteorological data based on your selected filters.")
        st.dataframe(meteorological_filtered_df, use_container_width=True)
    else:
        st.info("No historical meteorological data available for the current filters. Please adjust your selections in the sidebar.")

# --- Function to display Volcanic Activity Analysis Section ---
def display_volcanic_activity_analysis(df):
    st.header("Volcanic Activity Analysis")
    st.markdown("Examines volcanic gas flux data to monitor activity levels.")

    if df.empty:
        st.error("Cannot proceed with Volcanic Activity Analysis as no data was loaded.")
        return

    # --- Sidebar Filters for Volcanic Activity ---
    st.sidebar.header("Filter Volcanic Activity Data")
    selected_volcanic_location = st.sidebar.multiselect(
        "Select Volcanic Location(s):",
        options=df['Location'].unique(),
        default=df['Location'].unique(),
        key='volcanic_location_filter'
    )
    volcanic_filtered_df = df[df['Location'].isin(selected_volcanic_location)]

    if not volcanic_filtered_df.empty:
        volcanic_min_date = volcanic_filtered_df['Date'].min().to_pydatetime()
        volcanic_max_date = volcanic_filtered_df['Date'].max().to_pydatetime()
        volcanic_date_range = st.sidebar.slider(
            "Select Volcanic Date Range:",
            value=(volcanic_min_date, volcanic_max_date),
            format="YYYY-MM-DD",
            key='volcanic_date_range_filter'
        )
        volcanic_filtered_df = volcanic_filtered_df[(volcanic_filtered_df['Date'] >= pd.to_datetime(volcanic_date_range[0])) &
                                                    (volcanic_filtered_df['Date'] <= pd.to_datetime(volcanic_date_range[1]))]
    else:
        st.warning("No volcanic activity data available for the selected locations to determine date range.")
        volcanic_filtered_df = pd.DataFrame()


    # --- Volcanic Activity Parameter Trends ---
    st.subheader("Parameter Trends")
    volcanic_parameter_options = [col for col in df.columns if col not in ['Date', 'Location']]
    if not volcanic_parameter_options:
        st.warning("No measurable volcanic parameters found in the loaded data.")
        selected_volcanic_parameter = None
    else:
        selected_volcanic_parameter = st.selectbox("Choose a volcanic parameter to analyze:", volcanic_parameter_options, key='select_volcanic_param')

    if selected_volcanic_parameter and not volcanic_filtered_df.empty:
        st.write(f'Trend of {selected_volcanic_parameter} Over Time by Location')
        try:
            line_chart_data = volcanic_filtered_df.pivot_table(index='Date', columns='Location', values=selected_volcanic_parameter, aggfunc='mean').fillna(0)
            st.line_chart(line_chart_data)
        except Exception as e:
            st.warning(f"Could not render volcanic line chart. Please adjust filters or check data. Error: {e}")
            st.dataframe(volcanic_filtered_df[['Date', 'Location', selected_volcanic_parameter]])
    elif selected_volcanic_parameter:
        st.info("No volcanic activity data available for the selected filters for plotting trends.")
    else:
        if not df.empty:
            st.info("Please select a volcanic parameter to view trends.")

    st.markdown("---")

    # --- Volcanic Activity Summary Statistics ---
    st.subheader("Summary Statistics")
    if not volcanic_filtered_df.empty and volcanic_parameter_options:
        st.dataframe(volcanic_filtered_df.groupby('Location')[volcanic_parameter_options].describe().T.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}), use_container_width=True)
    else:
        st.info("Apply filters and select a volcanic parameter to see summary statistics.")

    st.markdown("---")

    # --- Volcanic Activity Parameter Distribution (Bar Chart of Averages) ---
    st.subheader("Parameter Distribution (Averages)")
    if selected_volcanic_parameter and not volcanic_filtered_df.empty:
        st.write(f'Average {selected_volcanic_parameter} by Location')
        avg_data = volcanic_filtered_df.groupby('Location')[selected_volcanic_parameter].mean().reset_index()
        st.bar_chart(avg_data.set_index('Location'))
    elif selected_volcanic_parameter:
        st.info("No volcanic activity data available for the selected filters for plotting distribution.")
    else:
        if not df.empty:
            st.info("Please select a volcanic parameter to view trends.")

    st.markdown("---")

    # --- Volcanic Activity Historical Data Section ---
    st.subheader("Historical Data")
    if not volcanic_filtered_df.empty:
        st.write("Below is a table displaying the historical volcanic activity data based on your selected filters.")
        st.dataframe(volcanic_filtered_df, use_container_width=True)
    else:
        st.info("No historical volcanic activity data available for the current filters. Please adjust your selections in the sidebar.")


# --- Main Application Logic ---

# Initialize session state for current page and selected data type
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Introduction' # Default page
if 'selected_data_type' not in st.session_state:
    st.session_state['selected_data_type'] = None # Default no data type selected
if 'selected_pollutant_display' not in st.session_state:
    st.session_state['selected_pollutant_display'] = 'Water Quality Parameters' # Default for pollutant levels display options

# Sidebar for navigation
st.sidebar.title("Navigation")

# Button for Introduction
if st.sidebar.button("Introduction"):
    st.session_state['current_page'] = 'Introduction'
    st.session_state['selected_data_type'] = None # Reset data type when intro is selected
    st.session_state['selected_pollutant_display'] = 'Water Quality Parameters' # Reset to default for pollutant section

# Renamed button for Historical Data Analysis
if st.sidebar.button("Historical Data Analysis"):
    st.session_state['current_page'] = 'Historical Data Analysis'
    # Default to Water Quality if Historical Data Analysis is newly selected and no sub-type yet
    if st.session_state['selected_data_type'] is None:
        st.session_state['selected_data_type'] = 'Water Quality'
    st.session_state['selected_pollutant_display'] = 'Water Quality Parameters' # Reset to default for pollutant section

# New button for Historical Pollutant Levels
if st.sidebar.button("Historical Pollutant Levels"):
    st.session_state['current_page'] = 'Historical Pollutant Levels'
    st.session_state['selected_data_type'] = None # Reset selected data type to avoid conflicts
    # Default to 'Water Quality Parameters' when this button is pressed
    if st.session_state['selected_pollutant_display'] is None:
        st.session_state['selected_pollutant_display'] = 'Water Quality Parameters'

# New button for WQI Prediction Tool
if st.sidebar.button("Predict WQI"):
    st.session_state['current_page'] = 'Predict WQI'
    st.session_state['selected_data_type'] = None # Reset data type
    st.session_state['selected_pollutant_display'] = 'Water Quality Parameters' # Reset to default for pollutant section


# Display content based on selected page
if st.session_state['current_page'] == 'Introduction':
    display_introduction()
elif st.session_state['current_page'] == 'Historical Pollutant Levels':
    display_pollutant_levels_analysis(water_df, meteorological_df, volcanic_df)
elif st.session_state['current_page'] == 'Predict WQI':
    display_wqi_prediction_tool(water_df)
else: # 'Historical Data Analysis' page is selected
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Data Type")
    
    # Radio buttons for selecting data type
    data_type_options = ['Water Quality', 'Meteorological', 'Volcanic Activity']
    st.session_state['selected_data_type'] = st.sidebar.radio(
        "Choose a data type:",
        data_type_options,
        index=data_type_options.index(st.session_state['selected_data_type']) if st.session_state['selected_data_type'] else 0,
        key='data_type_radio'
    )

    if st.session_state['selected_data_type'] == 'Water Quality':
        display_water_quality_analysis(water_df)
    elif st.session_state['selected_data_type'] == 'Meteorological':
        display_meteorological_analysis(meteorological_df)
    elif st.session_state['selected_data_type'] == 'Volcanic Activity':
        display_volcanic_activity_analysis(volcanic_df)

# --- Overall Disclaimer ---
st.markdown("""
<br>
<small><i>Note: This dashboard is now loaded with data from 'water_quality_filled (1).csv', 'completed_meteorological_filled.csv', and 'completed_volcanic_flux_filled.csv'. Ensure the CSV columns match the expected parameters for accurate visualization.</i></small>
""", unsafe_allow_html=True)
