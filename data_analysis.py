import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

import plotly.express as px
import plotly.graph_objects as go

# Setting basic page information
st.set_page_config(page_title="NBA Shot Analysis", layout="wide")
st.title("NBA Shot Quality Analysis Dashboard")

# Loading the data
@st.cache_data
def load_data():
    df = pd.read_csv("shot_logs.csv")
    # Dropping missing value rows
    df = df.dropna()
    
    # Converting the shot result to a binary value
    if df['SHOT_RESULT'].dtype == 'object':
        df['SHOT_RESULT'] = df['SHOT_RESULT'].map({'made': 1, 'missed': 0})
    
    return df

# Preprocessing the data for predictions with the neural network
@st.cache_data
def preprocess_for_prediction(df):
    df_processed = df.copy()
    
    # Converting the game clock to seconds remaining in the quarter
    df_processed['GAME_CLOCK'] = df_processed['GAME_CLOCK'].apply(
        lambda x: int(x.split(':')[0]) * 60 + int(x.split(':')[1]) if isinstance(x, str) else x
    )
    
    # Encoding the categorical variables using Label Encoding
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    # Drop the unnecessary columns
    columns_to_drop = ['GAME_ID', 'MATCHUP', 'LOCATION', 'W', 'FGM', 'PTS', 
                      'FINAL_MARGIN', 'SHOT_NUMBER', 'PERIOD', 
                      'CLOSEST_DEFENDER_PLAYER_ID', 'CLOSEST_DEFENDER', 
                      'player_name', 'player_id', 'SHOT_RESULT']
    
    columns_to_drop = [col for col in columns_to_drop if col in df_processed.columns]
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    return df_processed

# Loading the neural network model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('neural_network_model.pkl')
        return model
    except:
        st.error("Neural network model not found. Please ensure 'neural_network_model.pkl' is in the project directory.")
        return None

# Loading the data and model
try:
    df = load_data()
    nn_model = load_model()
except:
    st.error("Please ensure your shot data CSV file is in the correct location")
    st.stop()

# Adding a sidebar for filtering based on player
st.sidebar.header("Filters")
selected_player = st.sidebar.selectbox("Select Player", sorted(df['player_name'].unique()))

# Filtering data based on selections
filtered_df = df[
    (df['player_name'] == selected_player)
]

# Creating two columns for the layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Shot Distance vs Defender Distance")
    
    fig = go.Figure()

    # Scatter plot of shot distance vs defender distance
    fig.add_trace(go.Scatter(
        x=filtered_df['SHOT_DIST'],
        y=filtered_df['CLOSE_DEF_DIST'],
        mode='markers',
        marker=dict(
            size=8,
            color=filtered_df['SHOT_RESULT'],
            colorscale=['red', 'green'],
            colorbar=dict(title="Shot Made"),
        ),
        text=filtered_df.apply(
            lambda row: f"Shot Distance: {row['SHOT_DIST']}ft<br>"
                       f"Defender Distance: {row['CLOSE_DEF_DIST']}ft<br>"
                       f"Points: {row['PTS_TYPE']}<br>"
                       f"Shot Made: {'Yes' if row['SHOT_RESULT']==1 else 'No'}",
            axis=1
        ),
        hoverinfo='text',
        name='Shots'
    ))

    # Updating the layout
    fig.update_layout(
        width=600,
        height=500,
        xaxis_title="Shot Distance (ft)",
        yaxis_title="Closest Defender Distance (ft)",
        showlegend=True
    )

    st.plotly_chart(fig)

with col2:
    st.subheader("Shot Analysis")

    # Calculating statistics
    total_shots = len(filtered_df)
    made_shots = filtered_df['SHOT_RESULT'].sum()
    actual_fg_pct = made_shots / total_shots if total_shots > 0 else 0

    # Displaying the metrics
    col2_1, col2_2, col2_3 = st.columns(3)
    col2_1.metric("Total Shots", total_shots)
    col2_2.metric("Actual FG%", f"{actual_fg_pct:.1%}")
    col2_3.metric("Points per Shot", f"{filtered_df['PTS'].mean():.2f}")

    # Adding the distance distribution
    st.subheader("Shot Distance Distribution")
    fig_dist = px.histogram(
        filtered_df,
        x='SHOT_DIST',
        nbins=20,
        title='Shot Distance Distribution',
        color_discrete_sequence=['steelblue']
    )
    fig_dist.update_layout(
        xaxis_title="Shot Distance (ft)",
        yaxis_title="Count"
    )
    st.plotly_chart(fig_dist)

# Shot Quality Analysis Section
if nn_model is not None:
    st.subheader("Shot Quality Analysis (Neural Network Predictions)")
    
    try:
        # Preprocess the filtered data for prediction
        prediction_data = preprocess_for_prediction(filtered_df)
        
        # Get shot probabilities from the neural network
        shot_probabilities = nn_model.predict_proba(prediction_data)[:, 1]
        
        # Calculate expected points (P(make) * point value)
        expected_points = shot_probabilities * filtered_df['PTS_TYPE'].values
        
        quality_df = filtered_df.copy()
        quality_df['shot_probability'] = shot_probabilities
        quality_df['expected_points'] = expected_points
        
        # Create three columns for shot quality metrics
        qual_col1, qual_col2, qual_col3 = st.columns(3)
        
        with qual_col1:
            st.metric("Avg Shot Probability", f"{shot_probabilities.mean():.1%}")
            st.metric("Expected Points per Shot", f"{expected_points.mean():.2f}")
        
        with qual_col2:
            # Showing the distribution of expected points
            fig_exp_pts = px.histogram(
                quality_df,
                x='expected_points',
                nbins=15,
                title='Expected Points Distribution',
                color_discrete_sequence=['orange']
            )
            fig_exp_pts.update_layout(
                xaxis_title="Expected Points",
                yaxis_title="Count"
            )
            st.plotly_chart(fig_exp_pts)
        
        with qual_col3:
            # Showing the shot quality vs distance
            fig_quality = go.Figure()
            fig_quality.add_trace(go.Scatter(
                x=quality_df['SHOT_DIST'],
                y=quality_df['shot_probability'],
                mode='markers',
                marker=dict(
                    size=8,
                    color=quality_df['expected_points'],
                    colorscale='Viridis',
                    colorbar=dict(title="Expected Points"),
                ),
                text=quality_df.apply(
                    lambda row: f"Distance: {row['SHOT_DIST']}ft<br>"
                               f"Probability: {row['shot_probability']:.1%}<br>"
                               f"Expected Points: {row['expected_points']:.2f}",
                    axis=1
                ),
                hoverinfo='text',
                name='Shot Quality'
            ))
            
            fig_quality.update_layout(
                title="Shot Quality vs Distance",
                xaxis_title="Shot Distance (ft)",
                yaxis_title="Shot Probability",
                height=400
            )
            st.plotly_chart(fig_quality)
        
        st.subheader("Shot Quality Insights")
        
        # Categorizing the shots by quality
        quality_df['quality_category'] = pd.cut(
            quality_df['expected_points'], 
            bins=[0, 0.8, 1.2, float('inf')], 
            labels=['Poor', 'Average', 'Excellent']
        )
        
        qual_insights_col1, qual_insights_col2 = st.columns(2)
        
        with qual_insights_col1:
            # Quality distribution pie chart
            quality_dist = quality_df['quality_category'].value_counts()
            fig_quality_pie = px.pie(
                values=quality_dist.values,
                names=quality_dist.index,
                title="Shot Quality Distribution",
                color_discrete_map={'Poor': 'red', 'Average': 'yellow', 'Excellent': 'green'}
            )
            st.plotly_chart(fig_quality_pie)
        
        with qual_insights_col2:
            # Quality by shot type
            quality_by_type = quality_df.groupby('PTS_TYPE').agg({
                'shot_probability': 'mean',
                'expected_points': 'mean',
                'SHOT_RESULT': 'count'
            }).reset_index()
            quality_by_type.columns = ['Shot_Type', 'Avg_Probability', 'Avg_Expected_Points', 'Shot_Count']
            
            fig_quality_type = go.Figure()
            fig_quality_type.add_trace(go.Bar(
                x=quality_by_type['Shot_Type'],
                y=quality_by_type['Avg_Expected_Points'],
                text=quality_by_type['Shot_Count'],
                texttemplate='%{text} shots',
                textposition='outside',
                name='Expected Points',
                marker_color='lightgreen'
            ))
            
            fig_quality_type.update_layout(
                title="Expected Points by Shot Type",
                xaxis_title="Shot Type",
                yaxis_title="Average Expected Points"
            )
            st.plotly_chart(fig_quality_type)
            
    except Exception as e:
        st.error(f"Error in shot quality analysis: {str(e)}")
        st.info("This might be due to differences in data preprocessing between training and prediction.")

with col2:
    # Adding efficiency by defender distance
    st.subheader("Shooting Efficiency vs Defender Distance")
    
    # Creating bins for defender distance
    filtered_df_copy = filtered_df.copy()
    filtered_df_copy['def_dist_bins'] = pd.cut(filtered_df_copy['CLOSE_DEF_DIST'], bins=5)
    
    defender_analysis = filtered_df_copy.groupby('def_dist_bins').agg({
        'SHOT_RESULT': ['mean', 'count']
    }).reset_index()
    
    defender_analysis.columns = ['def_dist_bins', 'fg_pct', 'shot_count']
    
    # Only show bins with at least 5 shots
    defender_analysis = defender_analysis[defender_analysis['shot_count'] >= 5]

    if len(defender_analysis) > 0:
        fig_def = go.Figure()
        fig_def.add_trace(go.Bar(
            x=defender_analysis['def_dist_bins'].astype(str),
            y=defender_analysis['fg_pct'],
            name='Field Goal %',
            marker_color='lightblue'
        ))
        
        fig_def.update_layout(
            xaxis_title="Defender Distance Range (ft)",
            yaxis_title="Field Goal %",
            yaxis=dict(tickformat='.1%')
        )
        
        st.plotly_chart(fig_def)
    else:
        st.write("Not enough data to show defender distance analysis")

# Shot type analysis
st.subheader("Shot Type Analysis")
col3, col4 = st.columns(2)

with col3:
    # Points distribution
    pts_dist = filtered_df['PTS_TYPE'].value_counts().sort_index()
    fig_pts = px.pie(
        values=pts_dist.values,
        names=[f"{int(x)}-pointer" for x in pts_dist.index],
        title="Shot Type Distribution"
    )
    st.plotly_chart(fig_pts)

with col4:
    # Efficiency by shot type
    shot_eff = filtered_df.groupby('PTS_TYPE').agg({
        'SHOT_RESULT': 'mean',
        'PTS': 'count'
    }).reset_index()
    shot_eff.columns = ['PTS_TYPE', 'FG_PCT', 'COUNT']
    
    fig_eff = px.bar(
        shot_eff,
        x='PTS_TYPE',
        y='FG_PCT',
        title="Field Goal % by Shot Type",
        text='COUNT'
    )
    fig_eff.update_traces(texttemplate='%{text} shots', textposition='outside')
    fig_eff.update_layout(
        xaxis_title="Shot Type",
        yaxis_title="Field Goal %",
        yaxis=dict(tickformat='.1%')
    )
    st.plotly_chart(fig_eff)