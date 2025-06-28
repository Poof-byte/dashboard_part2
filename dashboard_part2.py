import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration
st.set_page_config(layout="wide", page_title="Water Quality Analysis Dashboard")

# --- Introduction Section (from water-quality-intro) ---
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

# --- Data Generation (for demonstration purposes) ---
@st.cache_data
def generate_sample_data():
    """Generates sample water quality data."""
    data = {
        'Date': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='D')),
        'Location': ['Site A'] * 50 + ['Site B'] * 50,
        'pH': [7.0 + 0.5 * i % 10 - 0.2 * (i // 10) for i in range(100)],
        'Turbidity (NTU)': [5 + 2 * (i % 5) + (i // 20) for i in range(100)],
        'Conductivity (uS/cm)': [300 + 10 * (i % 10) - 5 * (i // 15) for i in range(100)],
        'Dissolved Oxygen (mg/L)': [8.0 - 0.3 * (i % 7) + 0.1 * (i // 12) for i in range(100)],
        'E.coli (CFU/100mL)': [10 + 5 * (i % 3) for i in range(100)]
    }
    df = pd.DataFrame(data)
    # Introduce some outliers for demonstration
    df.loc[15, 'pH'] = 6.0
    df.loc[70, 'Turbidity (NTU)'] = 25
    df.loc[30, 'E.coli (CFU/100mL)'] = 200
    return df

df = generate_sample_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Data")

selected_location = st.sidebar.multiselect(
    "Select Location(s):",
    options=df['Location'].unique(),
    default=df['Location'].unique()
)

filtered_df = df[df['Location'].isin(selected_location)]

# Date range slider
min_date = filtered_df['Date'].min().to_pydatetime()
max_date = filtered_df['Date'].max().to_pydatetime()
date_range = st.sidebar.slider(
    "Select Date Range:",
    value=(min_date, max_date),
    format="YYYY-MM-DD"
)
filtered_df = filtered_df[(filtered_df['Date'] >= pd.to_datetime(date_range[0])) &
                          (filtered_df['Date'] <= pd.to_datetime(date_range[1]))]


# --- Main Dashboard Content ---
st.header("Water Quality Parameter Trends")

# Select parameter to display
parameter_options = ['pH', 'Turbidity (NTU)', 'Conductivity (uS/cm)', 'Dissolved Oxygen (mg/L)', 'E.coli (CFU/100mL)']
selected_parameter = st.selectbox("Choose a parameter to analyze:", parameter_options)

# Line chart for the selected parameter over time
if not filtered_df.empty:
    fig = px.line(
        filtered_df,
        x='Date',
        y=selected_parameter,
        color='Location',
        title=f'{selected_parameter} Over Time by Location',
        labels={'Date': 'Date', selected_parameter: selected_parameter, 'Location': 'Monitoring Location'},
        hover_name='Location',
        line_shape="linear"
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("No data available for the selected filters.")

st.markdown("---")

st.header("Summary Statistics")
if not filtered_df.empty:
    st.dataframe(filtered_df.groupby('Location')[parameter_options].describe().T.style.set_properties(**{'background-color': '#f0f2f6', 'color': 'black'}), use_container_width=True)
else:
    st.info("Apply filters to see summary statistics.")

st.markdown("---")

st.header("Parameter Distribution (Box Plot)")
if not filtered_df.empty:
    box_fig = px.box(
        filtered_df,
        x='Location',
        y=selected_parameter,
        title=f'Distribution of {selected_parameter} by Location',
        labels={'Location': 'Monitoring Location', selected_parameter: selected_parameter},
        color='Location'
    )
    st.plotly_chart(box_fig, use_container_width=True)
else:
    st.info("Apply filters to see parameter distribution.")

# --- Disclaimer ---
st.markdown("""
<br>
<small><i>Note: This dashboard uses randomly generated sample data for demonstration purposes. Real-world water quality data would require proper data acquisition and integration.</i></small>
""", unsafe_allow_html=True)

