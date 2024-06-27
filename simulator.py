import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import random
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import logging
import warnings
import scipy.stats as stats
import time

# Set page configuration
st.set_page_config(page_title='[24]7.ai Ads Simulator Test Bed', layout='wide')

# Suppress warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Custom CSS styles
custom_css = """
<style>
/* Add your custom CSS styles here */
body {
    font-family: Arial, sans-serif;
    line-height: 1.6;
}

.sidebar .sidebar-content {
    background-color: #f0f0f0;
    padding: 20px;
    border-radius: 10px;
}

.sidebar .sidebar-content h2 {
    color: #333;
    font-size: 20px;
    margin-bottom: 15px;
}

.sidebar .sidebar-content .stSlider {
    margin-bottom: 10px;
}

.sidebar .sidebar-content .stButton {
    margin-top: 15px;
    margin-right: 10px;
}

.sidebar .sidebar-content .stForm {
    margin-bottom: 20px;
}

.sidebar .sidebar-content .stMarkdown {
    margin-bottom: 15px;
}

.sidebar .sidebar-content .stMarkdown:last-child {
    margin-bottom: 0;
}

.sidebar .sidebar-content .stTextInput {
    width: 100%;
}

.sidebar .sidebar-content .stSelectbox {
    width: 100%;
}

.sidebar .sidebar-content .stMultiselect {
    width: 100%;
}

.sidebar .sidebar-content .stTextArea {
    width: 100%;
    height: 100px;
}

.sidebar .sidebar-content .stNumberInput {
    width: 100%;
}

.sidebar .sidebar-content .stColorPicker {
    width: 100%;
}

.sidebar .sidebar-content .stFileUploader {
    width: 100%;
}

.sidebar .sidebar-content .stButton>div {
    display: block;
    text-align: center;
}

.sidebar .sidebar-content .stMarkdown a {
    color: #0366d6;
    text-decoration: underline;
}

.sidebar .sidebar-content .stMarkdown a:hover {
    color: #033ea3;
}

.sidebar .sidebar-content .stMarkdown code {
    background-color: #f0f0f0;
    padding: 2px 4px;
    border-radius: 3px;
    font-size: 90%;
}

.content-section .stButton {
    margin-bottom: 10px;
}

.content-section .stMarkdown {
    margin-bottom: 20px;
}

.content-section .stMarkdown:last-child {
    margin-bottom: 0;
}

.content-section .stAlert {
    margin-bottom: 15px;
}

.content-section .stTable {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
}

.content-section .stTable th,
.content-section .stTable td {
    border: 1px solid #ddd;
    padding: 8px;
    text-align: left;
}

.content-section .stTable th {
    background-color: #f2f2f2;
    font-size: 14px;
    font-weight: normal;
    color: #333;
}

.content-section .stTable td {
    font-size: 14px;
    color: #333;
}

.content-section .stImage {
    margin-bottom: 20px;
}

.content-section .stGraph {
    margin-bottom: 20px;
}

.content-section .stPlotlyChart {
    margin-bottom: 20px;
}

.content-section .stDeckGlJsonChart {
    margin-bottom: 20px;
}

.content-section .stVegaLiteChart {
    margin-bottom: 20px;
}

.content-section .stVideo {
    margin-bottom: 20px;
}

.content-section .stAudio {
    margin-bottom: 20px;
}

.content-section .stFileUploader {
    margin-bottom: 20px;
}

.content-section .stProgress {
    margin-bottom: 20px;
}

.content-section .stTextArea {
    width: 100%;
    height: 100px;
}

.content-section .stTextInput {
    width: 100%;
}

.content-section .stNumberInput {
    width: 100%;
}

.content-section .stSelectbox {
    width: 100%;
}

.content-section .stMultiselect {
    width: 100%;
}

.content-section .stSlider {
    width: 100%;
}

.content-section .stDatePicker {
    width: 100%;
}

.content-section .stTimePicker {
    width: 100%;
}

.content-section .stColorPicker {
    width: 100%;
}

.content-section .stFileUploader {
    width: 100%;
}

.footer {
    background-color: #f0f0f0;
    padding: 10px 20px;
    text-align: center;
    border-top: 1px solid #ccc;
    margin-top: 20px;
    clear: both;
}

.footer p {
    margin: 0;
    font-size: 14px;
    color: #666;
}

.footer a {
    color: #0366d6;
    text-decoration: none;
}

.footer a:hover {
    text-decoration: underline;
}

</style>
"""

# Inject the CSS into the Streamlit app
st.markdown(custom_css, unsafe_allow_html=True)

# Function to simulate a single day
def simulate_day(df, cluster_affinities_df, models=None, num_ads=5, ts_params=None, threshold=0.5, model_type=None):
    ads_list = [x for x in range(num_ads)]

    for row in range(len(df)):
        if models:
            if model_type and 'adaptive_lr' in model_type:
                adaptive_lr_index = model_type.index('adaptive_lr')
                probabilities = [model.predict_proba(df.iloc[[row]][attributes])[0][1] for model in models if model is not None]
                if probabilities:
                    best_ad = None
                    for ad, prob in enumerate(probabilities):
                        if prob > threshold[adaptive_lr_index]:
                            best_ad = ad
                            break
                    if best_ad is None:
                        best_ad = random.choice(ads_list)
                else:
                    best_ad = random.choice(ads_list)
            else:
                probabilities = [model.predict_proba(df.iloc[[row]][attributes])[0][1] for model in models if model is not None]
                if probabilities:
                    best_ad = np.argmax(probabilities)
                else:
                    best_ad = random.choice(ads_list)
        elif ts_params:
            sampled_theta = [np.random.beta(ts_params[ad]['alpha'], ts_params[ad]['beta']) for ad in range(num_ads)]
            best_ad = np.argmax(sampled_theta)
        else:
            best_ad = random.choice(ads_list)

        df.at[row, 'Attached Ad'] = best_ad
        cluster_idx = df['Cluster'].iloc[row]
        if cluster_idx >= cluster_affinities_df.shape[0] or best_ad >= cluster_affinities_df.shape[1]:
            st.error(f"Index out of bounds: cluster_idx={cluster_idx}, best_ad={best_ad}")
            continue
        df.at[row, 'Affinity'] = round(float(cluster_affinities_df.iat[cluster_idx, best_ad]), 2)

        rand_prob = round(random.random(), 2)
        df.at[row, 'rand_prob'] = rand_prob

        if rand_prob < df.at[row, 'Affinity']:
            df.at[row, 'Random Clicked'] = 1
            if ts_params:
                ts_params[best_ad]['alpha'] += 1
        else:
            df.at[row, 'Random Clicked'] = 0
            if ts_params:
                ts_params[best_ad]['beta'] += 1

    return df, ts_params

# Function to calculate CTR for the day
def calculate_ctr(df):
    return df['Random Clicked'].sum() / len(df) if 'Random Clicked' in df else 0

# Function to calculate cluster-wise CTR
def calculate_clusterwise_ctr(df, num_clusters):
    return {cluster: calculate_ctr(df[df['Cluster'] == cluster]) for cluster in range(num_clusters)}

# Function to simulate the first day (random ad serving)
def simulate_first_day(df, cluster_affinities_df, num_ads=5):
    ads_list = list(range(num_ads))
    for row in range(len(df)):
        best_ad = random.choice(ads_list)  # Random ad serving
        df.at[row, 'Attached Ad'] = best_ad
        cluster_idx = df['Cluster'].iloc[row]
        if cluster_idx >= cluster_affinities_df.shape[0] or best_ad >= cluster_affinities_df.shape[1]:
            st.error(f"Index out of bounds: cluster_idx={cluster_idx}, best_ad={best_ad}")
            continue
        df.at[row, 'Affinity'] = round(float(cluster_affinities_df.iat[cluster_idx, best_ad]), 2)

        rand_prob = round(random.random(), 2)
        df.at[row, 'rand_prob'] = rand_prob
        df.at[row, 'Random Clicked'] = 1 if rand_prob < df.at[row, 'Affinity'] else 0

    return df

# Function to train models
def train_models(df_day, attributes, num_ads, model_type):
    models = []
    train_dfs = [df_day[df_day['Attached Ad'] == ad] for ad in range(num_ads)]
    for train_df in train_dfs:
        if len(train_df['Random Clicked'].unique()) > 1:
            if model_type == 'lr':
                model = LogisticRegression()
            model.fit(train_df[attributes], train_df['Random Clicked'])
            models.append(model)
        else:
            models.append(None)
    return models

# Initialize the Streamlit app
st.title('[24]7.ai Ads Simulator Test Bed')

# Tabs for simulation and animation
tab1, tab2 = st.tabs(["Simulation", "Animation"])

with tab1:
    st.sidebar.header('Simulation Parameters')

    # Sidebar inputs
    d = st.sidebar.slider('Dimension (d)', min_value=1, max_value=25, value=10)
    n = st.sidebar.slider('Initial Number of People ', min_value=100, max_value=100000, value=1000)
    num_ads = st.sidebar.slider('Number of Ads ', min_value=1, max_value=10, value=5)
    num_days = st.sidebar.slider('Number of Days ', min_value=2, max_value=30, value=7)
    samples_per_day_min, samples_per_day_max = st.sidebar.slider('Expected new engagements (range):', min_value=50, max_value=10000, value=(500, 500))
    models_to_compare = st.sidebar.multiselect('Models to Compare', options=['lr', 'random', 'adaptive_lr_1', 'adaptive_lr_2', 'ts'], default=['lr', 'random', 'adaptive_lr_1', 'adaptive_lr_2', 'ts'])
    adaptive_threshold_1 = st.sidebar.slider('Adaptive Threshold 1', min_value=0.0, max_value=1.0, value=0.5)
    adaptive_threshold_2 = st.sidebar.slider('Adaptive Threshold 2', min_value=0.0, max_value=1.0, value=0.7)

    # Ask for the number of geographic locations
    num_locations = st.sidebar.slider('Number of Geographic Locations', min_value=2, max_value=10, value=4)

    # Automatically generate geographic locations based on the number of clusters
    locations = {f"Location_{i+1}": (random.uniform(-90, 90), random.uniform(-180, 180)) for i in range(num_locations)}

    # Update the affinity matrix if the number of locations or ads changes
    if 'num_locations' not in st.session_state or st.session_state['num_locations'] != num_locations or 'num_ads' not in st.session_state or st.session_state['num_ads'] != num_ads:
        st.session_state['num_locations'] = num_locations
        st.session_state['num_ads'] = num_ads
        default_data = np.round(np.random.rand(num_locations, num_ads), 2).tolist()
        st.session_state['affinity_matrix_df'] = pd.DataFrame(default_data, columns=[f"Ad_{i+1}" for i in range(num_ads)], index=[f"Location_{i+1}" for i in range(num_locations)])


        # Customize the affinity matrix display
    st.markdown("### Affinity Matrix")
    with st.form(key='affinity_form'):
        affinity_matrix_df = st.session_state['affinity_matrix_df'].reset_index().rename(columns={'index': 'Location'})
        gb = GridOptionsBuilder.from_dataframe(affinity_matrix_df)
        gb.configure_default_column(editable=True)
        grid_options = gb.build()

        grid_response = AgGrid(affinity_matrix_df, gridOptions=grid_options, update_mode=GridUpdateMode.VALUE_CHANGED, height=150, width='100%')
        st.session_state['affinity_matrix_df'] = pd.DataFrame(grid_response['data']).set_index('Location')
        cluster_affinities_df = st.session_state['affinity_matrix_df']

        submit_button = st.form_submit_button(label='Submit Affinity Matrix')
        if submit_button:
            st.success("Affinity Matrix updated successfully!")


    # Initialize placeholders for results and data
    if 'all_data' not in st.session_state:
        st.session_state['all_data'] = {model: pd.DataFrame() for model in models_to_compare}
    if 'iteration_ctrs' not in st.session_state:
        st.session_state['iteration_ctrs'] = {model: [] for model in models_to_compare}
    if 'cluster_info' not in st.session_state:
        st.session_state['cluster_info'] = {model: [] for model in models_to_compare}
    if 'log_data' not in st.session_state:
        st.session_state['log_data'] = {model: [] for model in models_to_compare}
    if 'ads_served_per_day' not in st.session_state:
        st.session_state['ads_served_per_day'] = {model: [] for model in models_to_compare}
    if 'ads_clicked_per_day' not in st.session_state:
        st.session_state['ads_clicked_per_day'] = {model: [] for model in models_to_compare}
    if 'cluster_ctrs' not in st.session_state:
        st.session_state['cluster_ctrs'] = {model: {cluster: [] for cluster in range(num_locations)} for model in models_to_compare}

    # Ensure cluster_ctrs is correctly initialized for all clusters
    for model in models_to_compare:
        for cluster in range(num_locations):
            if cluster not in st.session_state['cluster_ctrs'][model]:
                st.session_state['cluster_ctrs'][model][cluster] = []

    ads_served_per_day = st.session_state['ads_served_per_day']
    ads_clicked_per_day = st.session_state['ads_clicked_per_day']
    cluster_info = st.session_state['cluster_info']

    # Placeholders for live updates
    live_plot = st.empty()
    comparison_chart_placeholder = st.empty()
    trend_chart_placeholder = st.empty()
    beta_plot_placeholders = [st.empty() for _ in range(num_ads)]
    summary_stats_placeholder = st.empty()
    ad_performance_placeholder = {cluster: st.empty() for cluster in range(num_locations)}

    # Cluster-specific placeholders
    cluster_plot_placeholders = {cluster: st.empty() for cluster in range(num_locations)}
    animation_placeholder = st.empty()

    # Initialize performance data storage for each cluster
    cluster_performance_data = {cluster: pd.DataFrame(columns=['Day'] + [f'Ad_{ad+1}_Served' for ad in range(num_ads)] + [f'Ad_{ad+1}_Clicked' for ad in range(num_ads)]) for cluster in range(num_locations)}

    # Start simulation button
    if st.sidebar.button('Start Simulation'):
        st.success("Simulation Started! Please wait for results...")

        # Generate initial random vectors representing people
        people = np.random.rand(n, d)

        # Creating column names dynamically based on the dimension
        attributes = [f"Attribute{i+1}" for i in range(d)]

        # Creating a DataFrame with dynamic column names
        df = pd.DataFrame(people, columns=attributes)

        # Apply K-means clustering to group people by geographic locations
        kmeans = KMeans(n_clusters=num_locations, n_init=10)
        kmeans.fit(people)

        # Get the cluster assignments and centroids (geographic locations)
        geographic_locations = kmeans.labels_
        centroids = kmeans.cluster_centers_

        # Map numeric clusters to geographic location names
        location_mapping = {i: loc_name for i, loc_name in enumerate(locations.keys())}
        df['Geographic_Location'] = [location_mapping[loc] for loc in geographic_locations]
        df['Cluster'] = geographic_locations

        # Initialize new columns in the DataFrame
        df['Attached Ad'] = -1
        df['Affinity'] = -1.0
        df['rand_prob'] = -1.0
        df['Random Clicked'] = -1
        df['Day'] = 1

        # List to store CTRs for each day
        iteration_ctrs = st.session_state['iteration_ctrs']

        # Initialize Thompson Sampling parameters
        ts_params = {ad: {'alpha': 1, 'beta': 1} for ad in range(num_ads)}

        # Simulate the first day with random ad serving
        df_day1 = simulate_first_day(df.copy(), cluster_affinities_df, num_ads=num_ads)
        df_day1['Random Clicked'] = df_day1['Random Clicked'].fillna(0)  # Ensure no NaN values
        ctr_day1 = calculate_ctr(df_day1)

        # Log day 1 results
        log_data = st.session_state['log_data']
        for model in models_to_compare:
            log_data[model].append(df_day1.copy())

        all_data = st.session_state['all_data']
        for model in models_to_compare:
            iteration_ctrs[model].append(ctr_day1)
            all_data[model] = df_day1.copy()

        # Display initial results after day 1
        fig = go.Figure()
        for model in models_to_compare:
            fig.add_trace(go.Scatter(
                x=[1],
                y=iteration_ctrs[model],
                mode='lines+markers',
                name=f'CTR ({model.upper()})',
                text=[f"Day 1: Ads Served: {len(all_data[model][all_data[model]['Day'] == 1])}, Ads Clicked: {all_data[model][all_data[model]['Day'] == 1]['Random Clicked'].sum()}"],
                hoverinfo='text'
            ))

        fig.update_layout(
            title='CTR Variation Over Days (Model Comparison)',
            xaxis_title='Day',
            yaxis_title='CTR',
            legend_title='Models',
            hovermode='x unified',
            width=1000,
            height=600,
            template='plotly_dark'  # Apply dark theme
        )

        live_plot.plotly_chart(fig, use_container_width=True)

        # Train models
        models = {'lr': [], 'adaptive_lr_1': [], 'adaptive_lr_2': []}
        if 'lr' in models_to_compare:
            models['lr'] = train_models(df_day1, attributes, num_ads, 'lr')

        if 'adaptive_lr_1' in models_to_compare:
            models['adaptive_lr_1'] = train_models(df_day1, attributes, num_ads, 'lr')

        if 'adaptive_lr_2' in models_to_compare:
            models['adaptive_lr_2'] = train_models(df_day1, attributes, num_ads, 'lr')

        # Simulate multiple days
        for day in range(2, num_days + 1):
            # Generate new data for the new day
            samples_per_day = random.randint(samples_per_day_min, samples_per_day_max)
            people = np.random.rand(samples_per_day, d)
            df = pd.DataFrame(people, columns=attributes)

            # Assign each person to a cluster based on the distance from the centroids
            clusters = kmeans.predict(people)
            df['Geographic_Location'] = [location_mapping[loc] for loc in clusters]
            df['Cluster'] = clusters
            df['Attached Ad'] = -1
            df['Affinity'] = -1.0
            df['rand_prob'] = -1.0
            df['Random Clicked'] = -1
            df['Day'] = day

            if 'adaptive_lr_1' in models_to_compare:
                df_day_adaptive_lr_1, _ = simulate_day(df.copy(), cluster_affinities_df, models=models['adaptive_lr_1'], num_ads=num_ads, threshold=[adaptive_threshold_1], model_type=['adaptive_lr'])
                df_day_adaptive_lr_1['Random Clicked'] = df_day_adaptive_lr_1['Random Clicked'].fillna(0)  # Ensure no NaN values
                ctr_adaptive_lr_1 = calculate_ctr(df_day_adaptive_lr_1)
                iteration_ctrs['adaptive_lr_1'].append(ctr_adaptive_lr_1)
                log_data['adaptive_lr_1'].append(df_day_adaptive_lr_1.copy())
                all_data['adaptive_lr_1'] = pd.concat([all_data['adaptive_lr_1'], df_day_adaptive_lr_1], ignore_index=True)

            if 'adaptive_lr_2' in models_to_compare:
                df_day_adaptive_lr_2, _ = simulate_day(df.copy(), cluster_affinities_df, models=models['adaptive_lr_2'], num_ads=num_ads, threshold=[adaptive_threshold_2], model_type=['adaptive_lr'])
                df_day_adaptive_lr_2['Random Clicked'] = df_day_adaptive_lr_2['Random Clicked'].fillna(0)  # Ensure no NaN values
                ctr_adaptive_lr_2 = calculate_ctr(df_day_adaptive_lr_2)
                iteration_ctrs['adaptive_lr_2'].append(ctr_adaptive_lr_2)
                log_data['adaptive_lr_2'].append(df_day_adaptive_lr_2.copy())
                all_data['adaptive_lr_2'] = pd.concat([all_data['adaptive_lr_2'], df_day_adaptive_lr_2], ignore_index=True)

            if 'lr' in models_to_compare:
                df_day_lr, _ = simulate_day(df.copy(), cluster_affinities_df, models=models['lr'], num_ads=num_ads)
                df_day_lr['Random Clicked'] = df_day_lr['Random Clicked'].fillna(0)  # Ensure no NaN values
                ctr_lr = calculate_ctr(df_day_lr)
                iteration_ctrs['lr'].append(ctr_lr)
                log_data['lr'].append(df_day_lr.copy())
                all_data['lr'] = pd.concat([all_data['lr'], df_day_lr], ignore_index=True)

            if 'random' in models_to_compare:
                df_day_random, _ = simulate_day(df.copy(), cluster_affinities_df, num_ads=num_ads)
                df_day_random['Random Clicked'] = df_day_random['Random Clicked'].fillna(0)  # Ensure no NaN values
                ctr_random = calculate_ctr(df_day_random)
                iteration_ctrs['random'].append(ctr_random)
                log_data['random'].append(df_day_random.copy())
                all_data['random'] = pd.concat([all_data['random'], df_day_random], ignore_index=True)

            if 'ts' in models_to_compare:
                df_day_ts, ts_params = simulate_day(df.copy(), cluster_affinities_df, num_ads=num_ads, ts_params=ts_params)
                df_day_ts['Random Clicked'] = df_day_ts['Random Clicked'].fillna(0)  # Ensure no NaN values
                ctr_ts = calculate_ctr(df_day_ts)
                iteration_ctrs['ts'].append(ctr_ts)
                log_data['ts'].append(df_day_ts.copy())
                all_data['ts'] = pd.concat([all_data['ts'], df_day_ts], ignore_index=True)

            for model in models_to_compare:
                ads_served_per_day[model].append(all_data[model][all_data[model]['Day'] == day]['Attached Ad'].count())
                ads_clicked_per_day[model].append(all_data[model][all_data[model]['Day'] == day]['Random Clicked'].sum())

                # Update cluster-wise information
                cluster_summary = []
                for cluster in range(num_locations):
                    cluster_data = all_data[model][(all_data[model]['Cluster'] == cluster) & (all_data[model]['Day'] == day)]
                    ads_served = len(cluster_data)
                    ads_clicked = cluster_data['Random Clicked'].sum()
                    ad_served_count = cluster_data['Attached Ad'].value_counts().to_dict()
                    ad_clicked_count = cluster_data[cluster_data['Random Clicked'] == 1]['Attached Ad'].value_counts().to_dict()

                    cluster_summary.append({
                        'Cluster': cluster,
                        'Ads Served': ads_served,
                        'Ads Clicked': ads_clicked,
                        **{f'Ad {ad} Served': ad_served_count.get(ad, 0) for ad in range(num_ads)},
                        **{f'Ad {ad} Clicked': ad_clicked_count.get(ad, 0) for ad in range(num_ads)}
                    })

                cluster_info[model].append(cluster_summary)

            # Calculate and update cluster-wise CTRs
            for model in models_to_compare:
                cluster_ctrs = calculate_clusterwise_ctr(all_data[model][all_data[model]['Day'] == day], num_locations)
                for cluster, ctr in cluster_ctrs.items():
                    st.session_state['cluster_ctrs'][model][cluster].append(ctr)

            # Retrain models using all data collected so far
            if 'lr' in models_to_compare:
                models['lr'] = train_models(all_data['lr'], attributes, num_ads, 'lr')

            if 'adaptive_lr_1' in models_to_compare:
                models['adaptive_lr_1'] = train_models(all_data['adaptive_lr_1'], attributes, num_ads, 'lr')

            if 'adaptive_lr_2' in models_to_compare:
                models['adaptive_lr_2'] = train_models(all_data['adaptive_lr_2'], attributes, num_ads, 'lr')

            # Live update plots and tables
            fig = go.Figure()
            for model in models_to_compare:
                fig.add_trace(go.Scatter(
                    x=list(range(1, day + 1)),  # Include day 1
                    y=iteration_ctrs[model],
                    mode='lines+markers',
                    name=f'CTR ({model.upper()})',
                    text=[f"Day {d}: Ads Served: {len(all_data[model][all_data[model]['Day'] == d])}, Ads Clicked: {all_data[model][all_data[model]['Day'] == d]['Random Clicked'].sum()}" for d in range(1, day + 1)],
                    hoverinfo='text'
                ))

            fig.update_layout(
                title='CTR Variation Over Days (Model Comparison)',
                xaxis_title='Day',
                yaxis_title='CTR',
                legend_title='Models',
                hovermode='x unified',
                width=1000,
                height=600,
                template='plotly_dark'  # Apply dark theme
            )

            live_plot.plotly_chart(fig, use_container_width=True)

            # Update overall summary statistics
            summary_stats = {
                'Total Ads Served': sum([len(data) for data in st.session_state['all_data'].values()]),
                'Total Ads Clicked': sum([data['Random Clicked'].sum() for data in st.session_state['all_data'].values() if 'Random Clicked' in data]),
                'Average CTR': np.mean([calculate_ctr(data) for data in st.session_state['all_data'].values()])
            }
            summary_stats_placeholder.write(summary_stats)

            # Update comparison charts
            comparison_data = pd.DataFrame({
                'Model': models_to_compare,
                'Average CTR': [np.mean(st.session_state['iteration_ctrs'][model]) for model in models_to_compare],  # Include the first day's CTR
                'Total Ads Served': [sum(ads_served_per_day[model]) for model in models_to_compare],
                'Total Ads Clicked': [sum(ads_clicked_per_day[model]) for model in models_to_compare]
            })

            fig_comparison = go.Figure()
            fig_comparison.add_trace(go.Bar(
                x=comparison_data['Model'],
                y=comparison_data['Average CTR'],
                name='Average CTR'
            ))
            fig_comparison.add_trace(go.Bar(
                x=comparison_data['Model'],
                y=comparison_data['Total Ads Served'],
                name='Total Ads Served'
            ))
            fig_comparison.add_trace(go.Bar(
                x=comparison_data['Model'],
                y=comparison_data['Total Ads Clicked'],
                name='Total Ads Clicked'
            ))

            fig_comparison.update_layout(
                title='Model Comparison',
                barmode='stack',
                xaxis_title='Model',
                yaxis_title='Count',
                width=700,
                height=500,
                template='plotly_dark'  # Apply dark theme
            )
            comparison_chart_placeholder.plotly_chart(fig_comparison)

            # Update ad performance data for each cluster when served using lr
            for cluster in range(num_locations):
                if 'lr' in models_to_compare:
                    cluster_data = [log_data['lr'][d][log_data['lr'][d]['Cluster'] == cluster] for d in range(day)]
                    ad_performance_data = pd.DataFrame({
                        'Day': list(range(1, day + 1)),
                        **{f'Ad_{ad+1}_Served': [sum(cluster_data[d]['Attached Ad'] == ad) for d in range(day)] for ad in range(num_ads)},
                        **{f'Ad_{ad+1}_Clicked': [sum((cluster_data[d]['Attached Ad'] == ad) & (cluster_data[d]['Random Clicked'] == 1)) for d in range(day)] for ad in range(num_ads)}
                    })

                    cluster_performance_data[cluster] = pd.concat([cluster_performance_data[cluster], ad_performance_data], ignore_index=True)

                    # Get affinities for the current cluster
                    affinities = cluster_affinities_df.iloc[cluster].tolist()
                    affinities_text = ", ".join([f"{affinity:.2f}" for i, affinity in enumerate(affinities)])

                    # Plot ad performance for the current cluster
                    fig_ad_performance = go.Figure()
                    for ad in range(num_ads):
                        fig_ad_performance.add_trace(go.Bar(
                            x=cluster_performance_data[cluster]['Day'],
                            y=cluster_performance_data[cluster][f'Ad_{ad+1}_Served'],
                            name=f'Ad {ad+1} Served'
                        ))
                        fig_ad_performance.add_trace(go.Bar(
                            x=cluster_performance_data[cluster]['Day'],
                            y=cluster_performance_data[cluster][f'Ad_{ad+1}_Clicked'],
                            name=f'Ad {ad+1}_Clicked'
                        ))

                    fig_ad_performance.update_layout(
                        title=f'Ad Performance Over Days - Location {cluster + 1} [{affinities_text}]',
                        barmode='group',
                        xaxis_title='Day',
                        yaxis_title='Count',
                        width=1000,
                        height=600,
                        template='plotly_dark'  # Apply dark theme
                    )
                    cluster_plot_placeholders[cluster].plotly_chart(fig_ad_performance, use_container_width=True)

    # Customizable Reports
    if st.session_state['all_data']:
        st.sidebar.header("Customizable Reports")
        report_model = st.sidebar.selectbox('Select Model for Report', models_to_compare)
        report_day = st.sidebar.selectbox('Select Day for Report', ['ALL'] + list(range(1, num_days + 1)))
        report_cluster = st.sidebar.selectbox('Select Cluster for Report', ['ALL'] + list(range(num_locations)))

        if report_model in models_to_compare:
            report_data = st.session_state['all_data'][report_model]
            if report_day != 'ALL':
                report_data = report_data[report_data['Day'] == report_day]
            if report_cluster != 'ALL':
                report_data = report_data[report_data['Cluster'] == report_cluster]

            st.sidebar.download_button(
                label=f'Download {report_model.upper()} Report for Day {report_day} and Cluster {report_cluster} as CSV',
                data=report_data.to_csv(index=False).encode('utf-8'),
                file_name=f'simulation_report_{report_model}_day_{report_day}_cluster_{report_cluster}.csv',
                mime='text/csv'
            )

with tab2:
    st.header("Simulation Animation")
    st.write("Coming soon......")

    # Animation parameters
    num_frames = 100
    interval = 0.2  # seconds

    # Function to create a frame
    def create_frame(day, all_data, cluster_affinities_df, num_clusters, num_ads):
        fig = go.Figure()

        # Get data for the current day
        df_day = all_data[all_data['Day'] == day]

        # Plot clusters as circles
        for cluster in range(num_clusters):
            cluster_data = df_day[df_day['Cluster'] == cluster]
            x = np.random.rand() * 10
            y = np.random.rand() * 10
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers+text',
                text=f"Cluster {cluster + 1}",
                textposition='top center',
                marker=dict(size=100, opacity=0.5, color='blue'),
                showlegend=False
            ))

            # Plot affinities below the cluster
            affinities = cluster_affinities_df.iloc[cluster].tolist()
            affinities_text = "<br>".join([f"Ad {i+1}: {round(affinity, 2):.2f}" for i, affinity in enumerate(affinities)])
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y - 1],
                mode='text',
                text=affinities_text,
                textposition='bottom center',
                showlegend=False
            ))

            # Plot ads served within the cluster
            for _, row in cluster_data.iterrows():
                ad_served = row['Attached Ad']
                clicked = row['Random Clicked']
                color = 'green' if clicked else 'red'
                fig.add_trace(go.Scatter(
                    x=[x + np.random.rand() - 0.5],
                    y=[y + np.random.rand() - 0.5],
                    mode='markers+text',
                    text=f"Ad {ad_served + 1}",
                    textposition='top center',
                    marker=dict(size=15, color=color),
                    showlegend=False
                ))

        fig.update_layout(
            title=f'Ad Simulation Animation - Day {day}',
            xaxis_title='',
            yaxis_title='',
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            width=1000,
            height=600,
            template='plotly_dark'  # Apply dark theme
        )
        return fig

    # Simulate the animation frames
    if 'all_data' in st.session_state:
        for day in range(1, num_days + 1):
            if 'random' in st.session_state['all_data']:
                if not st.session_state['all_data']['random'].empty:
                    st.write(f"Processing Day {day}")
                    frame = create_frame(day, st.session_state['all_data']['random'], cluster_affinities_df, num_locations, num_ads)
                    animation_placeholder.plotly_chart(frame)
                    time.sleep(interval)
            #     else:
            #         st.write("No data available for 'random' model.")
            # else:
            #     st.write("Model 'random' not found in session state.")
