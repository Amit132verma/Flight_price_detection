import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Flight Price Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 20px 0;
}
.metric-card {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">✈️ Flight Price Predictor</h1>', unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
    st.session_state.pipeline = None
    st.session_state.model_metrics = {}

@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the flight data"""
    try:
        # For demo purposes, create sample data
        # In production, replace this with: dataset = pd.read_csv('flight_data.csv')
        
        # Sample data creation (replace with actual data loading)
        np.random.seed(42)
        n_samples = 1000
        
        airlines = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'GoAir', 'Vistara']
        sources = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
        destinations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
        
        dataset = pd.DataFrame({
            'Airline': np.random.choice(airlines, n_samples),
            'Source': np.random.choice(sources, n_samples),
            'Destination': np.random.choice(destinations, n_samples),
            'Date_of_Journey': pd.date_range('2024-01-01', periods=n_samples, freq='D'),
            'Dep_Time': [f"{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}" for _ in range(n_samples)],
            'Arrival_Time': [f"{np.random.randint(0,24):02d}:{np.random.randint(0,60):02d}" for _ in range(n_samples)],
            'Duration': [f"{np.random.randint(1,15)}h {np.random.randint(0,60)}m" for _ in range(n_samples)],
            'Total_Stops': np.random.choice(['non-stop', '1 stop', '2 stops', '3 stops'], n_samples),
            'Additional_Info': np.random.choice(['No info', 'In-flight meal not included', 'Business class'], n_samples),
            'Price': np.random.normal(8000, 3000, n_samples)
        })
        
        # Ensure positive prices
        dataset['Price'] = np.abs(dataset['Price'])
        
        # Preprocessing steps
        dataset['day'] = pd.to_datetime(dataset['Date_of_Journey']).dt.day
        dataset['month'] = pd.to_datetime(dataset['Date_of_Journey']).dt.month
        dataset['year'] = pd.to_datetime(dataset['Date_of_Journey']).dt.year
        dataset.drop(columns=['Date_of_Journey'], inplace=True)
        
        # Departure time preprocessing
        dataset['Dep_hr'] = dataset['Dep_Time'].str.split(':', expand=True)[0].astype(float)
        dataset['Dep_Minz'] = dataset['Dep_Time'].str.split(':', expand=True)[1].astype(float)
        dataset.drop(columns=['Dep_Time'], inplace=True)
        
        # Arrival time preprocessing
        val_1 = dataset["Arrival_Time"].str.split(":", expand=True)
        dataset['Arrival_hr'] = val_1[0].astype(float)
        dataset['Arrival_minz'] = val_1[1].astype(float)
        dataset.drop(columns=['Arrival_Time'], inplace=True)
        
        # Duration preprocessing
        val_3 = dataset['Duration'].str.split(' ', expand=True)
        val_3[0] = val_3[0].str.replace('5m', '5h')
        hours = val_3[0].str.split('h', expand=True)[0].astype(float)
        val_3[1] = val_3[1].fillna('0m')
        minz = val_3[1].str.split('m', expand=True)[0].astype(float) / 60
        dataset['Total_Duration_hrs'] = hours + minz
        dataset.drop(columns=['Duration'], inplace=True)
        
        # Total stops preprocessing
        dataset['Total_Stops'] = dataset['Total_Stops'].str.replace('non-stop', '0')
        dataset['Total_Stops'] = dataset['Total_Stops'].str.split(" ", expand=True)[0].astype(float)
        
        # Remove outliers
        q1 = dataset['Price'].quantile(0.25)
        q3 = dataset['Price'].quantile(0.75)
        IQR = q3 - q1
        lower_bound = q1 - 1.5 * IQR
        upper_bound = q3 + 1.5 * IQR
        dataset = dataset[(dataset['Price'] >= lower_bound) & (dataset['Price'] <= upper_bound)]
        
        return dataset
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def train_model(dataset):
    """Train the flight price prediction model"""
    try:
        X = dataset.drop(columns=["Price"])
        y = dataset["Price"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
        
        categorical_features = ["Airline", "Source", "Destination", "Additional_Info"]
        numerical_features = ["day", "month", "year", "Total_Stops", "Dep_hr", "Dep_Minz", 
                            "Arrival_hr", "Arrival_minz", "Total_Duration_hrs"]
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", PowerTransformer(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
        # Use RandomForest as the default model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42)
        
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_train_pred = pipeline.predict(X_train)
        y_test_pred = pipeline.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        metrics = {
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        return pipeline, metrics, X_test, y_test, y_test_pred
    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None, None, None, None, None

def create_visualizations(dataset):
    """Create various visualizations for EDA"""
    
    # Price distribution by airline
    fig1 = px.box(dataset, x='Airline', y='Price', title='Price Distribution by Airline')
    fig1.update_xaxis(tickangle=45)
    
    # Price vs Total Stops
    grouped_data = dataset.groupby('Total_Stops')['Price'].mean().reset_index()
    fig2 = px.bar(grouped_data, x='Total_Stops', y='Price', title='Average Price by Number of Stops')
    
    # Price variation by month
    monthly_data = dataset.groupby('month')['Price'].mean().reset_index()
    fig3 = px.line(monthly_data, x='month', y='Price', title='Price Variation by Month')
    
    # Source-wise flight count
    source_count = dataset.groupby('Source').size().reset_index(name='Count')
    fig4 = px.bar(source_count, x='Source', y='Count', title='Number of Flights by Source')
    fig4.update_xaxis(tickangle=45)
    
    return fig1, fig2, fig3, fig4

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["🏠 Home", "📊 Data Analysis", "🔮 Price Prediction", "📈 Model Performance"])

# Load data
with st.spinner("Loading and preprocessing data..."):
    dataset = load_and_preprocess_data()

if dataset is None:
    st.error("Failed to load data. Please check your data file.")
    st.stop()

# Home Page
if page == "🏠 Home":
    st.markdown("## Welcome to the Flight Price Predictor!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Data Analysis</h3>
        <p>Explore flight data patterns and trends</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>🔮 Price Prediction</h3>
        <p>Predict flight prices for your journey</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>📈 Model Performance</h3>
        <p>View model accuracy and metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Dataset Overview")
    st.dataframe(dataset.head(), use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(dataset))
    with col2:
        st.metric("Average Price", f"₹{dataset['Price'].mean():.0f}")
    with col3:
        st.metric("Price Range", f"₹{dataset['Price'].min():.0f} - ₹{dataset['Price'].max():.0f}")
    with col4:
        st.metric("Airlines", dataset['Airline'].nunique())

# Data Analysis Page
elif page == "📊 Data Analysis":
    st.markdown("## 📊 Flight Data Analysis")
    
    # Create visualizations
    fig1, fig2, fig3, fig4 = create_visualizations(dataset)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Additional insights
    st.markdown("### Key Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Most Expensive Airlines:**")
        expensive_airlines = dataset.groupby('Airline')['Price'].mean().sort_values(ascending=False).head(3)
        for airline, price in expensive_airlines.items():
            st.write(f"• {airline}: ₹{price:.0f}")
    
    with col2:
        st.markdown("**Busiest Routes:**")
        busy_routes = dataset.groupby(['Source', 'Destination']).size().sort_values(ascending=False).head(3)
        for (source, dest), count in busy_routes.items():
            st.write(f"• {source} → {dest}: {count} flights")

# Price Prediction Page
elif page == "🔮 Price Prediction":
    st.markdown("## 🔮 Flight Price Prediction")
    
    # Train model if not already trained
    if not st.session_state.model_trained:
        with st.spinner("Training the prediction model..."):
            pipeline, metrics, X_test, y_test, y_test_pred = train_model(dataset)
            if pipeline is not None:
                st.session_state.pipeline = pipeline
                st.session_state.model_metrics = metrics
                st.session_state.model_trained = True
                st.success("Model trained successfully!")
    
    if st.session_state.model_trained:
        st.markdown("### Enter Flight Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            airline = st.selectbox("Airline", dataset['Airline'].unique())
            source = st.selectbox("Source City", dataset['Source'].unique())
            destination = st.selectbox("Destination City", dataset['Destination'].unique())
            additional_info = st.selectbox("Additional Info", dataset['Additional_Info'].unique())
        
        with col2:
            day = st.slider("Day", 1, 31, 15)
            month = st.slider("Month", 1, 12, 6)
            year = st.slider("Year", 2024, 2026, 2024)
            total_stops = st.selectbox("Total Stops", [0, 1, 2, 3])
        
        with col3:
            dep_hr = st.slider("Departure Hour", 0, 23, 10)
            dep_minz = st.slider("Departure Minutes", 0, 59, 0)
            arrival_hr = st.slider("Arrival Hour", 0, 23, 14)
            arrival_minz = st.slider("Arrival Minutes", 0, 59, 30)
        
        # Calculate duration
        total_duration_hrs = (arrival_hr + arrival_minz/60) - (dep_hr + dep_minz/60)
        if total_duration_hrs < 0:
            total_duration_hrs += 24
        
        st.info(f"Calculated Flight Duration: {total_duration_hrs:.2f} hours")
        
        if st.button("🔍 Predict Price", type="primary"):
            # Create prediction data
            new_data = pd.DataFrame({
                "Airline": [airline],
                "Source": [source],
                "Destination": [destination],
                "day": [day],
                "month": [month],
                "year": [year],
                "Total_Stops": [total_stops],
                "Dep_hr": [float(dep_hr)],
                "Dep_Minz": [float(dep_minz)],
                "Arrival_hr": [float(arrival_hr)],
                "Arrival_minz": [float(arrival_minz)],
                "Total_Duration_hrs": [total_duration_hrs],
                "Additional_Info": [additional_info]
            })
            
            try:
                predicted_price = st.session_state.pipeline.predict(new_data)
                
                st.markdown(f"""
                <div class="prediction-box">
                <h3>✈️ Predicted Flight Price</h3>
                <h1 style="color: #1f77b4; font-size: 3rem;">₹{predicted_price[0]:.0f}</h1>
                <p>Based on the provided flight details</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Price category
                if predicted_price[0] < 5000:
                    category = "💰 Budget"
                    color = "green"
                elif predicted_price[0] < 10000:
                    category = "💳 Moderate"
                    color = "orange"
                else:
                    category = "💎 Premium"
                    color = "red"
                
                st.markdown(f"**Price Category:** <span style='color: {color}'>{category}</span>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

# Model Performance Page
elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance")
    
    if not st.session_state.model_trained:
        with st.spinner("Training the model to show performance metrics..."):
            pipeline, metrics, X_test, y_test, y_test_pred = train_model(dataset)
            if pipeline is not None:
                st.session_state.pipeline = pipeline
                st.session_state.model_metrics = metrics
                st.session_state.model_trained = True
    
    if st.session_state.model_trained:
        metrics = st.session_state.model_metrics
        
        # Display metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Train R² Score", f"{metrics['train_r2']:.3f}")
        with col2:
            st.metric("Test R² Score", f"{metrics['test_r2']:.3f}")
        with col3:
            st.metric("Train MSE", f"{metrics['train_mse']:.0f}")
        with col4:
            st.metric("Test MSE", f"{metrics['test_mse']:.0f}")
        
        # Model interpretation
        st.markdown("### Model Interpretation")
        
        if metrics['test_r2'] > 0.8:
            st.success("🎯 Excellent model performance! The model explains more than 80% of the price variance.")
        elif metrics['test_r2'] > 0.6:
            st.info("👍 Good model performance! The model explains more than 60% of the price variance.")
        else:
            st.warning("⚠️ Moderate model performance. Consider feature engineering or trying different algorithms.")
        
        # Feature importance (if using tree-based model)
        try:
            # Re-train to get feature importance
            pipeline_temp, _, _, _, _ = train_model(dataset)
            if hasattr(pipeline_temp.named_steps['model'], 'feature_importances_'):
                feature_names = (pipeline_temp.named_steps['preprocessor']
                               .named_transformers_['num'].get_feature_names_out().tolist() +
                               pipeline_temp.named_steps['preprocessor']
                               .named_transformers_['cat'].get_feature_names_out().tolist())
                
                importances = pipeline_temp.named_steps['model'].feature_importances_
                feature_importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(10)
                
                st.markdown("### Top 10 Most Important Features")
                fig = px.bar(feature_importance_df, x='importance', y='feature', 
                           orientation='h', title='Feature Importance')
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Feature importance visualization not available for this model type.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<p>Flight Price Predictor • Built with Streamlit 🚀</p>
</div>
""", unsafe_allow_html=True)
