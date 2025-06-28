import streamlit as st
import pandas as pd
# Removed: import plotly.express as px

# Set page configuration
st.set_page_config(layout="wide", page_title="Environmental Data Analysis Dashboard")

# --- Data Loading and Preprocessing - Water Quality ---
@st.cache_data
def load_water_quality_data():
    """Loads and preprocesses the water quality data from the uploaded CSV."""
    try:
        df = pd.read_csv('water_quality_filled (1).csv')
        df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
        df.rename(columns={'LOCATION': 'Location'}, inplace=True)
        numeric_cols = ['SURFACE_WATER_TEMP', 'MIDDLE_WATER_TEMP', 'BOTTOM_WATER_TEMP',
                        'PH_LEVEL', 'AMMONIA', 'NITRATE_NITRITE', 'PHOSPHATE', 'DISSOLVED_OXYGEN']
        df = df[['Date', 'Location'] + [col for col in numeric_cols if col in df.columns]]
        return df
    except FileNotFoundError:
        st.error("Error: 'water_quality_filled (1).csv' not found. Please ensure the file is in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during water quality data loading or preprocessing: {e}")
        return pd.DataFrame()

water_df = load_water_quality_data()

# --- Data Loading and Preprocessing - Meteorological ---
@st.cache_data
def load_meteorological_data():
    """Loads and preprocesses the meteorological data from the uploaded CSV."""
    try:
        df = pd.read_csv('completed_meteorological_filled.csv')
        df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
        df.rename(columns={'LOCATION': 'Location'}, inplace=True)
        meteorological_numeric_cols = ['RAINFALL', 'TMAX', 'TMIN', 'RH', 'WIND SPEED', 'WIND DIRECTION']
        df = df[['Date', 'Location'] + [col for col in meteorological_numeric_cols if col in df.columns]]
        return df
    except FileNotFoundError:
        st.error("Error: 'completed_meteorological_filled.csv' not found. Please ensure the file is in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during meteorological data loading or preprocessing: {e}")
        return pd.DataFrame()

meteorological_df = load_meteorological_data()

# --- Data Loading and Preprocessing - Volcanic Flux ---
@st.cache_data
def load_volcanic_data():
    """Loads and preprocesses the volcanic flux data from the uploaded CSV."""
    try:
        df = pd.read_csv('completed_volcanic_flux_filled.csv')
        df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')
        df.rename(columns={'LOCATION': 'Location'}, inplace=True)
        volcanic_numeric_cols = ['CO2 FLUX', 'SO2 FLUX']
        df = df[['Date', 'Location'] + [col for col in volcanic_numeric_cols if col in df.columns]]
        return df
    except FileNotFoundError:
        st.error("Error: 'completed_volcanic_flux_filled.csv' not found. Please ensure the file is in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred during volcanic data loading or preprocessing: {e}")
        return pd.DataFrame()

volcanic_df = load_volcanic_data()


# --- Function to display the Introduction Section ---
def display_introduction():
    """Displays a short introduction to the dashboard."""
    st.markdown("""
    ## Welcome to the Environmental Data Analysis Dashboard!

    Explore vital environmental data including water quality, meteorological conditions, and volcanic activity. This dashboard provides interactive visualizations and insights to help understand trends and patterns in environmental monitoring.
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
    st.markdown("This section is currently blank as per user request.")
    # All content removed to make this section blank.
    # You can add content here later if needed.


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
            st.info("Please select a volcanic parameter to view distribution.")

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
    st.session_state['selected_pollutant_display'] = 'Water Quality Only' # Default for pollutant levels display options

# Sidebar for navigation
st.sidebar.title("Navigation")

# Button for Introduction
if st.sidebar.button("Introduction"):
    st.session_state['current_page'] = 'Introduction'
    st.session_state['selected_data_type'] = None # Reset data type when intro is selected
    st.session_state['selected_pollutant_display'] = None # Reset pollutant display option

# Renamed button for Historical Data Analysis
if st.sidebar.button("Historical Data Analysis"):
    st.session_state['current_page'] = 'Historical Data Analysis'
    # Default to Water Quality if Historical Data Analysis is newly selected and no sub-type yet
    if st.session_state['selected_data_type'] is None:
        st.session_state['selected_data_type'] = 'Water Quality'
    st.session_state['selected_pollutant_display'] = None # Reset pollutant display option

# New button for Historical Pollutant Levels
if st.sidebar.button("Historical Pollutant Levels"):
    st.session_state['current_page'] = 'Historical Pollutant Levels'
    st.session_state['selected_data_type'] = None # Reset selected data type to avoid conflicts
    # Default to 'Water Quality Only' when this button is pressed
    if st.session_state['selected_pollutant_display'] is None:
        st.session_state['selected_pollutant_display'] = 'Water Quality Only'


# Display content based on selected page
if st.session_state['current_page'] == 'Introduction':
    display_introduction()
elif st.session_state['current_page'] == 'Historical Pollutant Levels':
    # Pass all dataframes to the pollutant analysis function for combined views
    display_pollutant_levels_analysis(water_df, meteorological_df, volcanic_df)
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
