import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from backend import run_automl, analyze_dataset, recommend_algorithm
import time
import base64
from io import StringIO, BytesIO
from streamlit.components.v1 import html


# Configure page
st.set_page_config(
    page_title="AutoML Tool", 
    #page_icon="🤖", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
        .header {
            color: #2c3e50;
            font-weight: 700;
        }
    </style>
""", unsafe_allow_html=True)

# App header
st.title("Algorithm Identification from Dataset")
st.markdown("""
    Upload your dataset and let our automated machine learning pipeline analyze, 
    preprocess, and build the best model for your prediction task.
""")

# Initialize session state variables
if 'dataset_info' not in st.session_state:
    st.session_state.dataset_info = None
if 'model_results' not in st.session_state:
    st.session_state.model_results = None
if 'recommendation' not in st.session_state:
    st.session_state.recommendation = None

# Sidebar for file upload and settings
# In the sidebar section where the file uploader is defined, add a callback to reset session state
with st.sidebar:
    st.header("Upload & Settings")
    
    # Define a callback function to reset session state when a new file is uploaded
    def reset_session_state():
        st.session_state.dataset_info = None
        st.session_state.model_results = None
        st.session_state.recommendation = None
        st.session_state.df = None
        st.session_state.target_column = None
    
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV)", 
        type=["csv"],
        help="Upload a CSV file with your data",
        on_change=reset_session_state  # This will trigger when a new file is uploaded
    )
    
    if uploaded_file is not None:
        try:
            # Check if this is a different file than before
            if 'current_file' not in st.session_state or st.session_state.current_file != uploaded_file.name:
                reset_session_state()
                st.session_state.current_file = uploaded_file.name
            
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            # Rest of your existing code...
            
            # Show basic info
            st.markdown("**Dataset Info**")
            st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            # Select target column
            target_column = st.selectbox(
                "Select target column", 
                options=df.columns,
                index=len(df.columns)-1,
                help="Select the column you want to predict"
            )
            
            # Store in session state
            st.session_state.df = df
            st.session_state.target_column = target_column
            
            # Show sample data
            st.markdown("**Sample Data**")
            st.dataframe(df.head(3))
            
            # Analyze dataset button
            if st.button("Analyze Dataset"):
                with st.spinner("Analyzing dataset..."):
                    try:
                        st.session_state.dataset_info = analyze_dataset(df, target_column, verbose=False)
                        st.success("Dataset analysis completed!")
                    except Exception as e:
                        st.error(f"Error analyzing dataset: {e}")
            
        except Exception as e:
            st.error(f"Error loading file: {e}")
    else:
        st.info("Please upload a CSV file to get started")
        st.stop()

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["Data Analysis", "Model Training", "Results", "Feature Importance"])

with tab1:  # Data Analysis
    st.header("Data Analysis")
    
    if st.session_state.dataset_info is None:
        st.warning("Please click 'Analyze Dataset' in the sidebar to view analysis")
    else:
        dataset_info = st.session_state.dataset_info
        
        # Display dataset characteristics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Samples", dataset_info['num_samples'])
            st.metric("Features", dataset_info['num_features'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Numerical Features", dataset_info['num_numerical'])
            st.metric("Categorical Features", dataset_info['num_categorical'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Target Type", dataset_info['target_type'].capitalize())
            if dataset_info['target_type'] == "classification":
                st.metric("Class Imbalance", "Yes" if dataset_info['is_imbalanced'] else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Data quality metrics
        st.subheader("Data Quality")
        col4, col5, col6 = st.columns(3)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Missing Values", "Yes" if dataset_info['has_missing_values'] else "No")
            if dataset_info['has_missing_values']:
                st.metric("Missing Ratio", f"{dataset_info['missing_ratio']:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col5:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Outliers", "Yes" if dataset_info['has_outliers'] else "No")
            if dataset_info['has_outliers']:
                avg_outlier_ratio = np.mean(list(dataset_info['outlier_ratios'].values()))
                st.metric("Avg Outlier Ratio", f"{avg_outlier_ratio:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col6:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Skewed Features", "Yes" if dataset_info['has_skewed_features'] else "No")
            st.metric("High Dimensionality", "Yes" if dataset_info['high_dimensionality'] else "No")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Correlation analysis
        if 'correlations' in dataset_info and 'target_correlations' in dataset_info['correlations']:
            st.subheader("Feature Correlations with Target")
            
            target_corrs = dataset_info['correlations']['target_correlations']
            if len(target_corrs) > 0:
                # Create interactive bar chart
                fig = px.bar(
                    target_corrs.head(10),
                    x=target_corrs.head(10).index,
                    y=target_corrs.head(10).values,
                    labels={'x': 'Feature', 'y': 'Correlation'},
                    title="Top Features by Correlation with Target"
                )
                fig.update_layout(
                    xaxis_title="Feature",
                    yaxis_title="Correlation",
                    hovermode="x"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No strong correlations found with target variable")

with tab2:
    st.header("Model Training")
    
    if st.session_state.dataset_info is None:
        st.warning("Please analyze the dataset first from the 'Data Analysis' tab")
    else:
        dataset_info = st.session_state.dataset_info
        
        # Show algorithm recommendation
        st.subheader("Algorithm Recommendation")
        
        if st.session_state.recommendation is None:
            if st.button("Get Algorithm Recommendation"):
                with st.spinner("Determining best algorithm..."):
                    try:
                        recommended_algo, explanation = recommend_algorithm(dataset_info, verbose=False)
                        st.session_state.recommendation = (recommended_algo, explanation)
                        st.success("Recommendation ready!")
                    except Exception as e:
                        st.error(f"Error getting recommendation: {e}")
        
        if st.session_state.recommendation is not None:
            recommended_algo, explanation = st.session_state.recommendation
            
            st.markdown('<div class="feature-card">', unsafe_allow_html=True)
            st.markdown(f"**Recommended Algorithm:** `{recommended_algo}`")
            for line in explanation.split("\n")[2:]:  # Skip first two lines
                if line.strip():
                    st.markdown(f"- {line.strip()}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Start training button
            if st.button("🚀 Start Model Training", use_container_width=True):
                with st.spinner("Running AutoML pipeline..."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    try:
                        # Run the actual AutoML pipeline
                        model, metrics, dataset_info, interpretation = run_automl(
                            st.session_state.df, 
                            st.session_state.target_column
                        )
                        
                        # Store results in session state
                        st.session_state.model_results = {
                            'model': model,
                            'metrics': metrics,
                            'interpretation': interpretation
                        }
                        
                        progress_bar.empty()
                        status_text.success("Model training completed!")
                        st.rerun()
                    except Exception as e:
                        progress_bar.empty()
                        status_text.error(f"Model training failed: {e}")

with tab3:
    st.header("Model Results")
    
    if st.session_state.model_results is None:
        st.warning("Please train the model first from the 'Model Training' tab")
    else:
        model_results = st.session_state.model_results
        metrics = model_results.get('metrics', {})
        
        if not metrics:
            st.error("No metrics available in the model results")
        elif st.session_state.dataset_info['target_type'] == "classification":
            st.subheader("Classification Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Accuracy", f"{metrics.get('accuracy', 'N/A'):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("F1 Score", f"{metrics.get('f1_score', 'N/A'):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Precision", f"{metrics.get('precision', 'N/A'):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Recall", f"{metrics.get('recall', 'N/A'):.4f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
        else:  # regression
            st.subheader("Regression Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("RMSE", f"{metrics.get('rmse', 'N/A'):.4f}" if 'rmse' in metrics else "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("MSE", f"{metrics.get('mse', 'N/A'):.4f}" if 'mse' in metrics else "N/A")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("R² Score", f"{metrics.get('r2', 'N/A'):.4f}" if 'r2' in metrics else "N/A")
                st.markdown('</div>', unsafe_allow_html=True)

with tab4:
    st.header("Feature Importance")
    
    if st.session_state.model_results is None:
        st.warning("Please train the model first from the 'Model Training' tab")
    else:
        model_results = st.session_state.model_results
        interpretation = model_results.get('interpretation', {})
        
        if 'feature_importance' in interpretation:
            st.subheader("Feature Importance")
            
            # Prepare data for visualization
            features = [x[0] for x in interpretation['feature_importance']]
            importance = [x[1] for x in interpretation['feature_importance']]
            
            # Create interactive bar chart
            fig = px.bar(
                x=importance,
                y=features,
                orientation='h',
                labels={'x': 'Importance', 'y': 'Feature'},
                title="Feature Importance Scores"
            )
            fig.update_layout(
                yaxis={'categoryorder':'total ascending'},
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show key features
            st.subheader("Key Influential Features")
            if 'key_features' in interpretation:
                for i, feature in enumerate(interpretation['key_features'][:5], 1):
                    st.markdown(f"{i}. **{feature}**")
            else:
                st.info("No key features identified")
            
            # Cumulative importance plot
            if 'feature_importance' in interpretation:
                st.subheader("Cumulative Feature Importance")
                
                # Calculate cumulative importance
                cumulative_importance = np.cumsum(importance)
                
                # Create line plot
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=features,
                    y=cumulative_importance,
                    mode='lines+markers',
                    name='Cumulative Importance'
                ))
                
                # Add 80% threshold line
                fig.add_hline(
                    y=0.8,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="80% Threshold",
                    annotation_position="bottom right"
                )
                
                fig.update_layout(
                    xaxis_title="Features",
                    yaxis_title="Cumulative Importance",
                    hovermode="x"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Feature importance not available for this model type")

# Add download button for model (placeholder)
if st.session_state.model_results is not None:
    st.sidebar.download_button(
        label="Download Model",
        data="Model would be serialized here",
        file_name="automl_model.pkl",
        mime="application/octet-stream",
        help="Download the trained model for later use"
    )