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
    st.markdown("""
    ## Welcome to the Environmental Data Analysis Dashboard!

    This interactive dashboard is designed to provide comprehensive insights into various aspects of environmental data, focusing on water quality, meteorological conditions, and volcanic activity. Utilizing Streamlit, a powerful open-source app framework, we aim to make complex environmental data accessible and understandable for researchers, policymakers, and the public.

    **Why Environmental Monitoring Matters:**

    Access to clean and safe water is fundamental for human health, ecosystem integrity, and sustainable development. Understanding meteorological conditions is vital for agriculture, disaster preparedness, and climate studies. Assessing volcanic activity is crucial for hazard assessment and risk mitigation. Monitoring and analyzing these environmental parameters are crucial for:

    * **Protecting Public Health:** Identifying contaminants in water and assessing volcanic hazards.
    * **Environmental Conservation:** Assessing the health of aquatic ecosystems and understanding geological processes.
    * **Resource Management:** Informing decisions related to water treatment, allocation, and disaster preparedness, as well as agricultural planning and climate impact assessment.

    **What You Can Explore:**

    This dashboard will allow you to:

    * Visualize key environmental parameters over time and across different locations.
    * Identify trends and anomalies in data.
    * Compare data against established standards and guidelines (where applicable).
    * Gain a deeper understanding of the factors influencing environmental health.

    We hope this tool empowers you with valuable insights into vital environmental resources and phenomena.
    """)

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
        water_filtered_df = pd.DataFrame() # Ensure filtered_df is empty if no locations selected


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
        st.write("Below is a table displaying the historical water quality data based on your selected filters.")
        st.dataframe(water_filtered_df, use_container_width=True)
    else:
        st.info("No historical water quality data available for the current filters. Please adjust your selections in the sidebar.")

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

# Initialize session state for current page
if 'current_page' not in st.session_state:
    st.session_state['current_page'] = 'Introduction' # Default page

# Sidebar for navigation
st.sidebar.title("Navigation")
if st.sidebar.button("Introduction"):
    st.session_state['current_page'] = 'Introduction'
if st.sidebar.button("Water Quality Data"):
    st.session_state['current_page'] = 'Water Quality'
if st.sidebar.button("Meteorological Data"):
    st.session_state['current_page'] = 'Meteorological'
if st.sidebar.button("Volcanic Activity Data"):
    st.session_state['current_page'] = 'Volcanic Activity'

# Display content based on selected page
if st.session_state['current_page'] == 'Introduction':
    display_introduction()
elif st.session_state['current_page'] == 'Water Quality':
    display_water_quality_analysis(water_df)
elif st.session_state['current_page'] == 'Meteorological':
    display_meteorological_analysis(meteorological_df)
elif st.session_state['current_page'] == 'Volcanic Activity':
    display_volcanic_activity_analysis(volcanic_df)

# --- Overall Disclaimer ---
st.markdown("""
<br>
<small><i>Note: This dashboard is now loaded with data from 'water_quality_filled (1).csv', 'completed_meteorological_filled.csv', and 'completed_volcanic_flux_filled.csv'. Ensure the CSV columns match the expected parameters for accurate visualization.</i></small>
""", unsafe_allow_html=True)
