import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
        # Create sample data for demonstration
        np.random.seed(42)
        n_samples = 1000
        
        airlines = ['IndiGo', 'Air India', 'Jet Airways', 'SpiceJet', 'GoAir', 'Vistara']
        sources = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
        destinations = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad', 'Pune', 'Ahmedabad']
        additional_info_options = ['No info', 'In-flight meal not included', 'Business class', 'Extra legroom']
        
        # Generate base prices with some logic
        base_prices = []
        airlines_data = []
        sources_data = []
        destinations_data = []
        stops_data = []
        duration_data = []
        additional_info_data = []
        dep_hr_data = []
        arrival_hr_data = []
        
        for i in range(n_samples):
            airline = np.random.choice(airlines)
            source = np.random.choice(sources)
            destination = np.random.choice([d for d in destinations if d != source])
            stops = np.random.choice([0, 1, 2, 3], p=[0.4, 0.35, 0.2, 0.05])
            
            # Base price calculation with some logic
            base_price = 3000
            
            # Airline premium
            if airline in ['Vistara', 'Air India']:
                base_price += 2000
            elif airline in ['IndiGo', 'SpiceJet']:
                base_price += 500
            
            # Route premium
            if source in ['Mumbai', 'Delhi'] or destination in ['Mumbai', 'Delhi']:
                base_price += 1500
            
            # Stops penalty
            base_price += stops * 800
            
            # Duration
            duration = 1.5 + stops * 0.5 + np.random.normal(0, 0.5)
            duration = max(1.0, duration)
            base_price += duration * 500
            
            # Time of day effect
            dep_hr = np.random.randint(5, 23)
            arrival_hr = int(dep_hr + duration) % 24
            
            if dep_hr in [6, 7, 8, 18, 19, 20]:  # Peak hours
                base_price += 1000
            
            # Additional info
            additional_info = np.random.choice(additional_info_options)
            if additional_info == 'Business class':
                base_price *= 2.5
            elif additional_info == 'Extra legroom':
                base_price += 1500
            
            # Add some randomness
            final_price = base_price + np.random.normal(0, 500)
            final_price = max(2000, final_price)  # Minimum price
            
            base_prices.append(final_price)
            airlines_data.append(airline)
            sources_data.append(source)
            destinations_data.append(destination)
            stops_data.append(stops)
            duration_data.append(duration)
            additional_info_data.append(additional_info)
            dep_hr_data.append(dep_hr)
            arrival_hr_data.append(arrival_hr)
        
        dataset = pd.DataFrame({
            'Airline': airlines_data,
            'Source': sources_data,
            'Destination': destinations_data,
            'day': np.random.randint(1, 32, n_samples),
            'month': np.random.randint(1, 13, n_samples),
            'year': np.random.choice([2024, 2025], n_samples),
            'Total_Stops': stops_data,
            'Dep_hr': dep_hr_data,
            'Dep_Minz': np.random.randint(0, 60, n_samples),
            'Arrival_hr': arrival_hr_data,
            'Arrival_minz': np.random.randint(0, 60, n_samples),
            'Total_Duration_hrs': duration_data,
            'Additional_Info': additional_info_data,
            'Price': base_prices
        })
        
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
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ]
        )
        
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
            destination = st.selectbox("Destination City", 
                                     [d for d in dataset['Destination'].unique() if d != source])
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
                
                # Show price breakdown
                st.markdown("### Price Factors")
                factors = []
                if total_stops > 0:
                    factors.append(f"• {total_stops} stop(s): +₹{total_stops * 800}")
                if airline in ['Vistara', 'Air India']:
                    factors.append(f"• Premium airline ({airline}): +₹2000")
                if additional_info == 'Business class':
                    factors.append("• Business class: +150% premium")
                if dep_hr in [6, 7, 8, 18, 19, 20]:
                    factors.append("• Peak hour departure: +₹1000")
                
                if factors:
                    for factor in factors:
                        st.write(factor)
                else:
                    st.write("• Base pricing applied")
                
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
        
        # Feature importance visualization
        try:
            # Get feature names after preprocessing
            pipeline_temp = st.session_state.pipeline
            
            # Get feature names
            num_features = pipeline_temp.named_steps['preprocessor'].named_transformers_['num'].get_feature_names_out()
            cat_features = pipeline_temp.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
            all_features = list(num_features) + list(cat_features)
            
            importances = pipeline_temp.named_steps['model'].feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': all_features,
                'importance': importances
            }).sort_values('importance', ascending=False).head(15)
            
            st.markdown("### Top 15 Most Important Features")
            fig = px.bar(feature_importance_df, x='importance', y='feature', 
                       orientation='h', title='Feature Importance')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.info("Feature importance visualization not available.")
        
        # Prediction vs Actual scatter plot
        st.markdown("### Model Predictions vs Actual Prices")
        
        # Create scatter plot for a subset of predictions
        try:
            sample_size = min(200, len(y_test))
            sample_indices = np.random.choice(len(y_test), sample_size, replace=False)
            
            scatter_df = pd.DataFrame({
                'Actual': y_test.iloc[sample_indices],
                'Predicted': y_test_pred[sample_indices]
            })
            
            fig_scatter = px.scatter(scatter_df, x='Actual', y='Predicted', 
                                   title=f'Actual vs Predicted Prices (Sample of {sample_size} flights)')
            
            # Add perfect prediction line
            min_val = min(scatter_df['Actual'].min(), scatter_df['Predicted'].min())
            max_val = max(scatter_df['Actual'].max(), scatter_df['Predicted'].max())
            fig_scatter.add_scatter(x=[min_val, max_val], y=[min_val, max_val], 
                                  mode='lines', name='Perfect Prediction', 
                                  line=dict(dash='dash', color='red'))
            
            st.plotly_chart(fig_scatter, use_container_width=True)
            
        except Exception as e:
            st.info("Prediction visualization not available.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
<p>Flight Price Predictor • Built with Streamlit 🚀</p>
<p>Upload your flight_data.csv file to use real data instead of sample data</p>
</div>
""", unsafe_allow_html=True)
