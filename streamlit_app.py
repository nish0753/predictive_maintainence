import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Manufacturing Quality Control Analysis",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 2rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 0.5rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown('<p class="main-header">🏭 Manufacturing Quality Control Analysis</p>', unsafe_allow_html=True)
st.markdown("### Predictive Maintenance Using Statistical Analysis & Machine Learning")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Project Overview", 
     "📊 Data Exploration", 
     "🤖 SVM Predictions", 
     "📉 Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Project by:** [Your Name]  
**Dataset:** Manufacturing Quality Control  
**Methods:** Minitab + SVM  
**Date:** November 2025
""")
# Load the trained model
@st.cache_resource(ttl=None)
def load_model():
    """Load the saved SVM model and preprocessing components"""
    model_path = Path("svm_model.pkl")
    
    if not model_path.exists():
        return None
    
    try:
        with open(model_path, "rb") as file:
            model_data = pickle.load(file)
        # Add model info for debugging
        st.sidebar.info(f"✅ Model loaded: {model_data.get('best_params', {})}")
        return model_data
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load dataset for exploration
@st.cache_data
def load_dataset():
    """Load the manufacturing dataset"""
    try:
        df = pd.read_csv("Manufacturing_dataset.xls")
        return df
    except:
        return None

model_data = load_model()
df = load_dataset()

# ============================================================================
# PAGE 1: PROJECT OVERVIEW
# ============================================================================
if page == "🏠 Project Overview":
    st.markdown('<p class="section-header">📋 Project Introduction</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Project Objective
        This project aims to **predict manufacturing quality and detect defects** in production processes 
        using a combination of **statistical analysis (Minitab)** and **machine learning (SVM)**.
        
        ### 📦 Dataset Overview
        The dataset contains manufacturing parameters collected from production lines, including:
        - **Process variables** (temperature, pressure, speed, etc.)
        - **Material properties**
        - **Equipment settings**
        - **Quality outcomes** (pass/fail or defect classification)
        
        ### 🔬 Methodology
        We employed a **dual approach**:
        
        1. **Statistical Analysis (Minitab)**
           - Hypothesis testing to identify significant factors
           - ANOVA to compare process variations
           - Understanding which variables affect quality
        
        2. **Machine Learning (SVM)**
           - Support Vector Machine for predictive modeling
           - Real-time defect prediction
           - Deployment for production use
        
        ### 💡 Key Insights
        - ✅ Identified critical process parameters affecting quality
        - ✅ Achieved high prediction accuracy for defect detection
        - ✅ Enabled proactive quality control and cost reduction
        - ✅ Provided actionable recommendations for process optimization
        """)
    
    with col2:
        st.info("""
        **📊 Quick Stats**
        
        🔹 Dataset Size: 1000+ samples  
        🔹 Features: Multiple process variables  
        🔹 Target: Quality classification  
        🔹 Accuracy: 85%+ (SVM)  
        🔹 Methods: Minitab + ML
        """)
        
        st.success("""
        **🎓 Learning Outcomes**
        
        ✓ Statistical hypothesis testing  
        ✓ Feature significance analysis  
        ✓ ML model development  
        ✓ Production deployment  
        ✓ Data-driven decision making
        """)
    
    st.markdown("---")
    st.markdown("### 🚀 How This Project Helps Manufacturing")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Proactive Quality Control**
        - Early defect detection
        - Reduced waste
        - Lower production costs
        """)
    
    with col2:
        st.markdown("""
        **📊 Data-Driven Decisions**
        - Evidence-based improvements
        - Process optimization
        - Root cause analysis
        """)
    
    with col3:
        st.markdown("""
        **⚡ Real-Time Predictions**
        - Instant quality assessment
        - Automated monitoring
        - Faster decision making
        """)

# ============================================================================
# PAGE 2: DATA EXPLORATION
# ============================================================================
elif page == "📊 Data Exploration":
    st.markdown('<p class="section-header">📊 Data Overview & Exploration</p>', unsafe_allow_html=True)
    
    if df is not None:
        # Dataset Summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📝 Total Rows", f"{len(df):,}")
        with col2:
            st.metric("📋 Total Columns", len(df.columns))
        with col3:
            st.metric("🔢 Numeric Features", len(df.select_dtypes(include=[np.number]).columns))
        with col4:
            st.metric("📊 Categorical Features", len(df.select_dtypes(exclude=[np.number]).columns))
        
        st.markdown("---")
        
        # Dataset Preview
        st.subheader("📄 Dataset Preview (First 10 Rows)")
        st.dataframe(df.head(10), use_container_width=True)
        
        st.markdown("---")
        
        # Summary Statistics
        st.subheader("📈 Summary Statistics")
        tab1, tab2 = st.tabs(["Numeric Features", "Dataset Info"])
        
        with tab1:
            st.dataframe(df.describe(), use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Types:**")
                st.write(df.dtypes)
            with col2:
                st.write("**Missing Values:**")
                missing = df.isnull().sum()
                if missing.sum() > 0:
                    st.write(missing[missing > 0])
                else:
                    st.success("✅ No missing values!")
        
        # Distribution Analysis
        st.markdown("---")
        st.subheader("📊 Feature Distribution")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_feature = st.selectbox("Select a feature to visualize:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.hist(df[selected_feature].dropna(), bins=30, color='steelblue', edgecolor='black')
                ax.set_xlabel(selected_feature)
                ax.set_ylabel('Frequency')
                ax.set_title(f'Distribution of {selected_feature}')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.boxplot(df[selected_feature].dropna(), vert=True)
                ax.set_ylabel(selected_feature)
                ax.set_title(f'Boxplot of {selected_feature}')
                st.pyplot(fig)

        # Control Charts (I-MR Style)
        st.markdown("---")
        st.subheader("📈 Process Control Charts (I-MR Style)")
        st.markdown("Monitor process stability over time (simulated sequence).")
        
        if numeric_cols:
            cc_feature = st.selectbox("Select feature for Control Chart:", numeric_cols, index=0, key='cc_select')
            
            # Prepare data: Use the balanced dataset logic (approx 2500 rows) to reduce noise
            target_col = 'Optimal Conditions'
            if target_col in df.columns:
                df0 = df[df[target_col] == 0]
                df1 = df[df[target_col] == 1]
                
                # Sample 1500 from Class 0 (matching training logic)
                if len(df0) > 1500:
                    df0_sample = df0.sample(n=1500, random_state=42)
                else:
                    df0_sample = df0
                
                # Combine
                df_balanced_viz = pd.concat([df0_sample, df1])
                
                # STRICT LIMIT: Sample exactly 1500 points total for cleaner visualization as requested
                if len(df_balanced_viz) > 1500:
                    df_balanced_viz = df_balanced_viz.sample(n=1500, random_state=42)
                
                # Sort by index to preserve time order
                df_balanced_viz = df_balanced_viz.sort_index()
                
                data_series = df_balanced_viz[cc_feature]
                st.caption(f"Visualizing Subset of Training Data (1,500 samples) - Sorted by Time")
            else:
                data_series = df[cc_feature].dropna().reset_index(drop=True)

            mean_val = data_series.mean()
            std_val = data_series.std()
            ucl = mean_val + 3 * std_val
            lcl = mean_val - 3 * std_val
            
            fig, ax = plt.subplots(figsize=(12, 5))
            # Plotting
            ax.plot(range(len(data_series)), data_series, marker='o', linestyle='-', markersize=2, color='#1f77b4', label='Data', alpha=0.6)
            ax.axhline(mean_val, color='green', linestyle='-', label='Mean')
            ax.axhline(ucl, color='red', linestyle='--', label='UCL (+3σ)')
            ax.axhline(lcl, color='red', linestyle='--', label='LCL (-3σ)')
            
            # Highlight out of control points
            outliers = data_series[(data_series > ucl) | (data_series < lcl)]
            ax.scatter(outliers.index.map(lambda x: df_balanced_viz.index.get_loc(x)), outliers, color='red', zorder=5, label='Out of Control')
            
            ax.set_title(f'Individual Chart for {cc_feature} (Balanced Data)')
            ax.set_xlabel('Observation Index')
            ax.set_ylabel(cc_feature)
            ax.legend(loc='upper right')
            st.pyplot(fig)
            
            st.info(f"""
            **Control Chart Statistics (Balanced Data):**
            - **Mean:** {mean_val:.4f}
            - **UCL (Upper Control Limit):** {ucl:.4f}
            - **LCL (Lower Control Limit):** {lcl:.4f}
            - **Out of Control Points:** {len(outliers)} ({len(outliers)/len(data_series):.1%})
            """)

        # Correlation Analysis
        st.markdown("---")
        st.subheader("🔥 Correlation Heatmap")
        st.markdown("Understand how different features are related to each other.")
        
        if len(numeric_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            corr = df[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)

        # Target Analysis
        st.markdown("---")
        st.subheader("🎯 Target Analysis (Optimal vs Not Optimal)")
        
        target_col = 'Optimal Conditions'
        if target_col in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Class Distribution (Pie Chart)**")
                fig, ax = plt.subplots(figsize=(6, 6))
                df[target_col].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#ff9999','#66b3ff'], labels=['Not Optimal (0)', 'Optimal (1)'])
                ax.set_ylabel('')
                st.pyplot(fig)
                
            with col2:
                st.markdown("**Class Distribution (Count Plot)**")
                fig, ax = plt.subplots(figsize=(6, 6))
                sns.countplot(x=target_col, data=df, ax=ax, palette='viridis')
                ax.set_xticklabels(['Not Optimal (0)', 'Optimal (1)'])
                st.pyplot(fig)
        
        # Bivariate Analysis
        st.markdown("---")
        st.subheader("🔗 Bivariate Analysis")
        st.markdown("Visualize the relationship between two features, colored by the target class.")
        
        if len(numeric_cols) >= 2:
            c1, c2 = st.columns(2)
            with c1:
                x_axis = st.selectbox("Select X-Axis Feature", numeric_cols, index=0)
            with c2:
                y_axis = st.selectbox("Select Y-Axis Feature", numeric_cols, index=min(1, len(numeric_cols)-1))
                
            fig, ax = plt.subplots(figsize=(10, 6))
            if target_col in df.columns:
                sns.scatterplot(data=df, x=x_axis, y=y_axis, hue=target_col, palette='deep', ax=ax, alpha=0.6)
                ax.legend(title='Condition', labels=['Not Optimal', 'Optimal'])
            else:
                sns.scatterplot(data=df, x=x_axis, y=y_axis, ax=ax)
            
            st.pyplot(fig)
    
    else:
        st.warning("⚠️ Dataset not found. Please ensure 'Manufacturing_dataset.xls' is in the app directory.")

# ============================================================================
# PAGE 3: SVM PREDICTIONS
# ============================================================================
elif page == "🤖 SVM Predictions":
    st.markdown('<p class="section-header">🤖 SVM Prediction Interface</p>', unsafe_allow_html=True)
    
    if model_data is not None:
        st.success("✅ Model loaded successfully!")
        
        st.markdown("""
        Use this interface to predict manufacturing quality based on process parameters.
        Enter the values below and click **Predict Quality** to get instant results.
        """)
        
        st.markdown("---")
        
        # Prediction Tabs
        tab1, tab2 = st.tabs(["🎯 Single Prediction", "📁 Batch Prediction"])
        
        # TAB 1: Manual Input
        with tab1:
            st.subheader("Enter Process Parameters")
            
            # Helper to load examples
            st.info("💡 **Need help?** Click a button below to load example values that result in **Optimal Conditions**:")
            
            # Define all examples
            examples = {
                "Example A": {'Temperature (°C)': 74.59, 'Machine Speed (RPM)': 1521.0, 'Production Quality Score': 8.04, 'Vibration Level (mm/s)': 0.05, 'Energy Consumption (kWh)': 1.12},
                "Example B": {'Temperature (°C)': 75.10, 'Machine Speed (RPM)': 1487.0, 'Production Quality Score': 8.97, 'Vibration Level (mm/s)': 0.05, 'Energy Consumption (kWh)': 1.05},
                "Example C": {'Temperature (°C)': 75.62, 'Machine Speed (RPM)': 1486.0, 'Production Quality Score': 8.20, 'Vibration Level (mm/s)': 0.06, 'Energy Consumption (kWh)': 1.39},
                "Example D": {'Temperature (°C)': 74.26, 'Machine Speed (RPM)': 1495.0, 'Production Quality Score': 8.08, 'Vibration Level (mm/s)': 0.03, 'Energy Consumption (kWh)': 1.99},
                "Example E": {'Temperature (°C)': 74.69, 'Machine Speed (RPM)': 1521.0, 'Production Quality Score': 8.96, 'Vibration Level (mm/s)': 0.05, 'Energy Consumption (kWh)': 1.57}
            }

            # Create columns for buttons
            cols = st.columns(len(examples))
            
            for idx, (name, values) in enumerate(examples.items()):
                with cols[idx]:
                    if st.button(f"Load {name}"):
                        for f_idx, feature in enumerate(model_data['features']):
                            if feature in values:
                                st.session_state[f"input_{f_idx}"] = values[feature]

            st.markdown("---")
            st.info("⚠️ **Need help?** Click a button below to load example values that result in **Not Optimal Conditions**:")

            # Define Not Optimal examples
            not_optimal_examples = {
                "Example X": {'Temperature (°C)': 75.65, 'Machine Speed (RPM)': 1518.0, 'Production Quality Score': 8.45, 'Vibration Level (mm/s)': 0.07, 'Energy Consumption (kWh)': 1.56},
                "Example Y": {'Temperature (°C)': 78.48, 'Machine Speed (RPM)': 1450.0, 'Production Quality Score': 8.99, 'Vibration Level (mm/s)': 0.07, 'Energy Consumption (kWh)': 1.28},
                "Example Z": {'Temperature (°C)': 75.11, 'Machine Speed (RPM)': 1462.0, 'Production Quality Score': 8.54, 'Vibration Level (mm/s)': 0.05, 'Energy Consumption (kWh)': 1.11},
                "Example W": {'Temperature (°C)': 75.38, 'Machine Speed (RPM)': 1483.0, 'Production Quality Score': 8.58, 'Vibration Level (mm/s)': 0.08, 'Energy Consumption (kWh)': 1.82},
                "Example V": {'Temperature (°C)': 73.94, 'Machine Speed (RPM)': 1470.0, 'Production Quality Score': 8.73, 'Vibration Level (mm/s)': 0.06, 'Energy Consumption (kWh)': 1.23}
            }

            # Create columns for Not Optimal buttons
            cols_no = st.columns(len(not_optimal_examples))
            
            for idx, (name, values) in enumerate(not_optimal_examples.items()):
                with cols_no[idx]:
                    if st.button(f"Load {name}"):
                        for f_idx, feature in enumerate(model_data['features']):
                            if feature in values:
                                st.session_state[f"input_{f_idx}"] = values[feature]
            
            # Create input form
            with st.form("prediction_form"):
                num_features = min(10, len(model_data['features']))
                
                # Create dynamic inputs based on feature names
                col1, col2 = st.columns(2)
                input_values = {}
                
                for idx, feature in enumerate(model_data['features'][:num_features]):
                    col = col1 if idx % 2 == 0 else col2
                    with col:
                        input_values[feature] = st.number_input(
                            f"{feature}",
                            value=0.0,
                            format="%.4f",
                            key=f"input_{idx}"
                        )
                
                submitted = st.form_submit_button("🔮 Predict Quality", use_container_width=True)
                
                if submitted:
                    with st.spinner("Analyzing..."):
                        try:
                            # Create dataframe
                            input_data = pd.DataFrame([input_values])
                            
                            # Add missing features
                            for feature in model_data['features']:
                                if feature not in input_data.columns:
                                    input_data[feature] = 0
                            
                            input_data = input_data[model_data['features']]
                            
                            # Scale and predict
                            X_scaled = model_data['scaler'].transform(input_data)
                            prediction = model_data['model'].predict(X_scaled)[0]
                            probability = model_data['model'].predict_proba(X_scaled)[0]
                            
                            # Decode prediction
                            if model_data['label_encoder'] is not None:
                                prediction_label = model_data['label_encoder'].inverse_transform([prediction])[0]
                            else:
                                # Fallback manual mapping if no encoder
                                prediction_label = "Optimal" if prediction == 1 else "Not Optimal"
                            
                            confidence = probability.max() * 100
                            
                            # Display result
                            st.markdown("---")
                            st.markdown("### 🎯 Prediction Result")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                if str(prediction_label) == "1" or str(prediction_label).lower() == "optimal":
                                    st.success(f"### ✅ Optimal Conditions")
                                else:
                                    st.error(f"### ❌ Not Optimal Conditions")
                            
                            with col2:
                                st.metric("Confidence", f"{confidence:.1f}%")
                                
                            with col3:
                                st.metric("Model", "SVM (RBF)")
                            
                            # Maintenance Recommendation Logic
                            st.markdown("### 🛠️ Maintenance Recommendation")
                            
                            is_optimal = str(prediction_label) == "1" or str(prediction_label).lower() == "optimal"
                            
                            if not is_optimal:
                                if confidence > 85:
                                    st.error("🔴 **CRITICAL ALERT:** High probability of failure. Stop machine and inspect immediately.")
                                else:
                                    st.warning("🟠 **WARNING:** Potential issue detected. Schedule maintenance inspection.")
                            else:
                                if confidence < 75:
                                    st.warning("🟡 **CAUTION:** Process is Optimal but unstable (Low Confidence). Monitor closely for drift.")
                                else:
                                    st.success("🟢 **HEALTHY:** System operating within normal parameters.")

                            # Show detailed probabilities
                            st.info(f"Probability of Optimal: {probability[1]*100:.1f}% | Probability of Not Optimal: {probability[0]*100:.1f}%")
                            
                            # Probability breakdown
                            st.markdown("#### 📊 Probability Distribution")
                            prob_df = pd.DataFrame({
                                'Class': model_data['label_encoder'].classes_ if model_data['label_encoder'] else range(len(probability)),
                                'Probability': probability
                            })
                            st.bar_chart(prob_df.set_index('Class'))
                            
                        except Exception as e:
                            st.error(f"Prediction error: {e}")
        
        # TAB 2: Batch Upload
        with tab2:
            st.subheader("Upload Data for Batch Predictions")
            st.info("Supports CSV and Excel files. If your file contains the target column 'Optimal Conditions', we will also calculate accuracy!")
            
            uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xls', 'xlsx'])
            
            if uploaded_file is not None:
                try:
                    # Determine file type and read accordingly
                    if uploaded_file.name.endswith('.csv'):
                        input_df = pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                        try:
                            input_df = pd.read_excel(uploaded_file)
                        except:
                            # Fallback: Try reading as CSV (sometimes .xls files are actually CSVs)
                            uploaded_file.seek(0)
                            input_df = pd.read_csv(uploaded_file)
                    
                    st.write(f"**Uploaded Data Preview ({len(input_df)} rows):**")
                    st.dataframe(input_df.head())
                    
                    if st.button("🔮 Predict All", use_container_width=True):
                        with st.spinner("Processing batch predictions..."):
                            # Preprocess
                            num_cols = input_df.select_dtypes(include=[np.number]).columns.tolist()
                            for col in num_cols:
                                if col in input_df.columns:
                                    input_df[col] = input_df[col].fillna(input_df[col].median())
                            
                            # Align features
                            for feature in model_data['features']:
                                if feature not in input_df.columns:
                                    input_df[feature] = 0
                            
                            X_processed = input_df[model_data['features']]
                            
                            # Predict
                            X_scaled = model_data['scaler'].transform(X_processed)
                            predictions = model_data['model'].predict(X_scaled)
                            probabilities = model_data['model'].predict_proba(X_scaled)
                            
                            # Decode
                            if model_data['label_encoder'] is not None:
                                predictions_decoded = model_data['label_encoder'].inverse_transform(predictions)
                            else:
                                predictions_decoded = predictions
                            
                            # Results
                            results_df = input_df.copy()
                            results_df['Predicted_Quality'] = predictions_decoded
                            results_df['Confidence'] = probabilities.max(axis=1) * 100
                            
                            # Highlight logic for display
                            def highlight_rows(row):
                                # Check if prediction is "Not Optimal" (assuming 0 or "Not Optimal")
                                is_not_optimal = str(row['Predicted_Quality']) == '0' or str(row['Predicted_Quality']).lower() == 'not optimal'
                                confidence = row['Confidence']
                                
                                # Red for Critical (Not Optimal + High Confidence)
                                if is_not_optimal and confidence > 85:
                                    return ['background-color: #ff4b4b; color: white'] * len(row) # Strong Red with White Text
                                # Yellow for Warning (Not Optimal + Low Confidence)
                                elif is_not_optimal:
                                    return ['background-color: #ffa726; color: black'] * len(row) # Orange/Yellow with Black Text
                                # Green for Optimal (Healthy)
                                else:
                                    return ['background-color: #d4edda; color: black'] * len(row) # Light Green

                            st.success("✅ Predictions completed!")
                            
                            # Apply styling to the dataframe display
                            st.dataframe(results_df.style.apply(highlight_rows, axis=1))
                            
                            # If Ground Truth exists, show metrics
                            target_col = 'Optimal Conditions' # Adjust if your target col name is different
                            if target_col in input_df.columns:
                                st.markdown("### 📊 Batch Performance Analysis")
                                y_true = input_df[target_col]
                                
                                # Handle encoding if y_true is not numeric/aligned
                                if model_data['label_encoder'] is not None:
                                    # Try to align types
                                    if y_true.dtype == object:
                                        pass # Assume it matches
                                else:
                                    pass

                                try:
                                    # Calculate Accuracy
                                    # We need to ensure y_true matches the format of predictions_decoded
                                    # If predictions are "Optimal"/"Not Optimal" and y_true is 0/1
                                    
                                    # Simple check: if predictions are strings and y_true is int
                                    y_pred_final = predictions_decoded
                                    
                                    # If we used manual mapping in the single prediction, we might need it here
                                    # But predictions_decoded comes from label_encoder or raw
                                    
                                    batch_acc = (y_true == predictions).mean() # Compare raw predictions (0/1) if possible
                                    if model_data['label_encoder']:
                                         # If we have an encoder, 'predictions' is the encoded (0/1) form usually? 
                                         # No, model.predict returns encoded labels.
                                         # So if y_true is 0/1, we can compare directly with 'predictions'
                                         batch_acc = (y_true == predictions).mean()
                                    else:
                                         batch_acc = (y_true == predictions).mean()

                                    st.metric("Batch Accuracy", f"{batch_acc:.2%}")
                                    
                                    # Confusion Matrix
                                    from sklearn.metrics import confusion_matrix
                                    cm = confusion_matrix(y_true, predictions)
                                    
                                    st.write("**Confusion Matrix:**")
                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                    ax.set_xlabel('Predicted')
                                    ax.set_ylabel('Actual')
                                    st.pyplot(fig)
                                    
                                except Exception as e:
                                    st.warning(f"Could not calculate accuracy: {e}")

                            st.dataframe(results_df)
                            
                            # Download
                            csv = results_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                "📥 Download Results",
                                csv,
                                "predictions.csv",
                                "text/csv",
                                use_container_width=True
                            )
                            
                            # Summary
                            st.markdown("#### 📊 Prediction Summary")
                            # Fix for Streamlit index error: Use simple Series value_counts
                            summary_counts = pd.Series(predictions_decoded).value_counts()
                            st.bar_chart(summary_counts)
                
                except Exception as e:
                    st.error(f"Error: {e}")
    
    else:
        st.error("⚠️ Model not found. Please train and save the model first using the notebook.")

# ============================================================================
# PAGE 5: MODEL PERFORMANCE
# ============================================================================
elif page == "📉 Model Performance":
    st.markdown('<p class="section-header">📉 Model Performance Metrics</p>', unsafe_allow_html=True)
    
    if model_data is not None:
        st.markdown("""
        Evaluation of the **Support Vector Machine (SVM)** model on test data.
        These metrics demonstrate the model's ability to predict quality accurately.
        """)
        
        st.markdown("---")
        
        # Performance Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🎯 Test Accuracy", f"{model_data['test_accuracy']:.2%}")
        with col2:
            st.metric("📊 CV Score", f"{model_data['cv_score']:.4f}")
        with col3:
            st.metric("🔧 Kernel", model_data['best_params']['kernel'].upper())
        with col4:
            st.metric("📝 Classes", model_data['n_classes'])
        
        st.markdown("---")
        
        # Model Details
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Model Configuration")
            st.json({
                "Algorithm": "Support Vector Machine (SVM)",
                "Kernel": model_data['best_params']['kernel'],
                "C (Regularization)": model_data['best_params']['C'],
                "Gamma": model_data['best_params']['gamma'],
                "Features": len(model_data['features']),
                "Classes": model_data['n_classes']
            })
        
        with col2:
            st.subheader("📊 Performance Summary")
            st.markdown(f"""
            - **Test Accuracy:** {model_data['test_accuracy']:.2%}
            - **Cross-Validation Score:** {model_data['cv_score']:.4f}
            - **Number of Features:** {len(model_data['features'])}
            - **Training Method:** Grid Search with 5-fold CV
            - **Best Parameters:** Automatically selected
            
            **Interpretation:**  
            The model achieved **{model_data['test_accuracy']:.1%}** accuracy on unseen test data,
            indicating {'excellent' if model_data['test_accuracy'] > 0.9 else 'good' if model_data['test_accuracy'] > 0.8 else 'fair'} 
            predictive performance for quality control.
            """)
        
        st.markdown("---")
        
        # Feature Importance Placeholder
        st.subheader("🔍 Top Features (by importance)")
        st.info("Note: Feature importance for SVM requires additional analysis. Top correlated features from data exploration can be displayed here.")
        
        if len(model_data['features']) > 0:
            st.write("**Features used in the model:**")
            features_df = pd.DataFrame({
                'Feature': model_data['features'][:10],
                'Index': range(min(10, len(model_data['features'])))
            })
            st.dataframe(features_df, use_container_width=True)
    
    else:
        st.error("⚠️ Model not found.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d;'>
    <p><strong>Manufacturing Quality Control Analysis</strong> | Powered by Minitab & SVM</p>
    <p>Built with Streamlit 🚀 | November 2025</p>
</div>
""", unsafe_allow_html=True)
