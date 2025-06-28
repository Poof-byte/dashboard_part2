import streamlit as st
import pandas as pd
# Removed: import plotly.express as px

# Set page configuration
st.set_page_config(layout="wide", page_title="Water Quality Analysis Dashboard")

# --- Introduction Section ---
st.markdown("""
## Welcome to the Water Quality Analysis Dashboard!

This interactive dashboard is designed to provide comprehensive insights into various aspects of water quality. Utilizing Streamlit, a powerful open-source app framework, we aim to make complex environmental data accessible and understandable for researchers, policymakers, and the public.

**Why Water Quality Matters:**

Access to clean and safe water is fundamental for human health, ecosystem integrity, and sustainable development. Monitoring and analyzing water quality parameters are crucial for:

* **Protecting Public Health:** Identifying contaminants and ensuring drinking water safety.

* **Environmental Conservation:** Assessing the health of aquatic ecosystems and preventing pollution.

* **Resource Management:** Informing decisions related to water treatment, allocation, and conservation efforts.

**What You Can Explore:**

This dashboard will allow you to:

* Visualize key water quality parameters over time and across different locations.

* Identify trends and anomalies in water data.

* Compare water quality against established standards and guidelines.

* Gain a deeper understanding of the factors influencing water health.

We hope this tool empowers you with valuable insights into the vital resource that is water.
""")

st.markdown("---") # Separator

# --- Data Loading and Preprocessing ---
@st.cache_data
def load_water_quality_data():
    """Loads and preprocesses the water quality data from the uploaded CSV."""
    try:
        # Load the CSV file
        df = pd.read_csv('water_quality_filled (1).csv')

        # Combine 'YEAR' and 'MONTH' to create a 'Date' column
        df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['MONTH'].astype(str) + '-01')

        # Rename 'LOCATION' to 'Location' for consistency with previous code
        df.rename(columns={'LOCATION': 'Location'}, inplace=True)

        # Select relevant columns for analysis
        # Ensure only numeric columns that are parameters are included for plotting
        numeric_cols = ['SURFACE_WATER_TEMP', 'MIDDLE_WATER_TEMP', 'BOTTOM_WATER_TEMP',
                        'PH_LEVEL', 'AMMONIA', 'NITRATE_NITRITE', 'PHOSPHATE', 'DISSOLVED_OXYGEN']
        
        # Filter for only relevant columns
        df = df[['Date', 'Location'] + [col for col in numeric_cols if col in df.columns]]

        return df
    except FileNotFoundError:
        st.error("Error: 'water_quality_filled (1).csv' not found. Please ensure the file is in the correct directory.")
        return pd.DataFrame() # Return an empty DataFrame on error
    except Exception as e:
        st.error(f"An error occurred during data loading or preprocessing: {e}")
        return pd.DataFrame()

df = load_water_quality_data()

# Check if data was loaded successfully
if df.empty:
    st.stop() # Stop the app if no data is loaded

# --- Sidebar Filters ---
st.sidebar.header("Filter Data")

# Get unique locations only if df is not empty
if not df.empty:
    selected_location = st.sidebar.multiselect(
        "Select Location(s):",
        options=df['Location'].unique(),
        default=df['Location'].unique()
    )
    # Filter by location first
    filtered_df = df[df['Location'].isin(selected_location)]

    # Date range slider
    # Ensure min_date and max_date are valid datetime objects
    if not filtered_df.empty:
        min_date = filtered_df['Date'].min().to_pydatetime()
        max_date = filtered_df['Date'].max().to_pydatetime()
        date_range = st.sidebar.slider(
            "Select Date Range:",
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(date_range[0])) &
                                  (filtered_df['Date'] <= pd.to_datetime(date_range[1]))]
    else:
        st.warning("No data available for the selected locations to determine date range.")
        filtered_df = pd.DataFrame() # Ensure filtered_df is empty if no locations selected
else:
    st.warning("No data loaded. Please check the CSV file.")
    filtered_df = pd.DataFrame()


# --- Main Dashboard Content ---
st.header("Water Quality Parameter Trends")

# Define parameter options based on the loaded CSV columns
# Exclude 'Date' and 'Location' from parameter options
if not df.empty:
    parameter_options = [col for col in df.columns if col not in ['Date', 'Location']]
    if not parameter_options:
        st.warning("No measurable water quality parameters found in the loaded data.")
        selected_parameter = None
    else:
        selected_parameter = st.selectbox("Choose a parameter to analyze:", parameter_options)
else:
    parameter_options = []
    selected_parameter = None


# Line chart for the selected parameter over time using Streamlit's native chart
if selected_parameter and not filtered_df.empty:
    st.subheader(f'{selected_parameter} Over Time by Location')
    
    try:
        # Create a pivot table to get locations as columns for st.line_chart
        # Fill NaN values that might arise from pivoting to ensure line_chart works smoothly
        line_chart_data = filtered_df.pivot_table(index='Date', columns='Location', values=selected_parameter, aggfunc='mean').fillna(0)
        st.line_chart(line_chart_data)
    except Exception as e:
        st.warning(f"Could not render line chart. Please adjust filters or check data. Error: {e}")
        st.dataframe(filtered_df[['Date', 'Location', selected_parameter]]) # Show raw data if chart fails
elif selected_parameter:
    st.warning("No data available for the selected filters for plotting trends.")
else:
    if not df.empty:
        st.info("Please select a parameter to view trends.")


st.markdown("---")

st.header("Summary Statistics")
if not filtered_df.empty and parameter_options:
    st.dataframe(filtered_df.groupby('Location')[parameter_options].describe().T.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}), use_container_width=True)
else:
    st.info("Apply filters and select a parameter to see summary statistics.")

st.markdown("---")

st.header("Parameter Distribution (Bar Chart of Averages)")
if selected_parameter and not filtered_df.empty:
    st.subheader(f'Average {selected_parameter} by Location')
    avg_data = filtered_df.groupby('Location')[selected_parameter].mean().reset_index()
    st.bar_chart(avg_data.set_index('Location'))
elif selected_parameter:
    st.info("No data available for the selected filters for plotting distribution.")
else:
    if not df.empty:
        st.info("Please select a parameter to view distribution.")

st.markdown("---")

# --- Water Quality Historical Data Section ---
st.header("Water Quality Historical Data")
if not filtered_df.empty:
    st.write("Below is a table displaying the historical water quality data based on your selected filters.")
    st.dataframe(filtered_df, use_container_width=True)
else:
    st.info("No historical data available for the current filters. Please adjust your selections in the sidebar.")


# --- Disclaimer ---
st.markdown("""
<br>
<small><i>Note: This dashboard is now loaded with data from 'water_quality_filled (1).csv'. Ensure the CSV columns match the expected parameters for accurate visualization.</i></small>
""", unsafe_allow_html=True)
