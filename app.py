import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import streamlit as st

# Streamlit page configuration
st.set_page_config(
    page_title="AgriSmart Crop Advisor 🌾",
    page_icon="🌱",
    layout="centered",
)

# Title and description
st.title("🌾 AgriSmart Crop Advisor 🌱")
st.write("""
Welcome to **AgriSmart Crop Advisor**! This app helps farmers and agricultural consultants predict the best crop to grow 
based on soil, location data, and farming practices. 🚜💡
""")

# Load and preprocess the dataset
@st.cache_data
def load_and_preprocess_data():
    # Load dataset directly from the script folder
    file_path = "Crop(Distric level).csv"  # Add your CSV file to the same directory
    data = pd.read_csv(file_path)
    
    # Handle missing values
    data = data.dropna()
    
    # Encode the target 'label' column
    label_encoder = LabelEncoder()
    data['label'] = label_encoder.fit_transform(data['label'])
    
    # One-hot encode the 'district' column
    data = pd.get_dummies(data, columns=['district'], drop_first=True)
    
    # Split features and target
    X = data.drop('label', axis=1)
    y = data['label']
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, label_encoder, X.columns

# Load model and label encoder
model, label_encoder, feature_columns = load_and_preprocess_data()

# User Input Interface
st.subheader("🌟 Enter Input Values for Crop Prediction 🌟")

# Input fields for user
col1, col2 = st.columns(2)

with col1:
    N = st.number_input("🌱 Nitrogen (N):", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
    P = st.number_input("🌿 Phosphorus (P):", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
    K = st.number_input("🍂 Potassium (K):", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
    ph = st.number_input("🧪 Soil pH:", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

with col2:
    temperature = st.number_input("🌡️ Temperature (°C):", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
    humidity = st.number_input("💧 Humidity (%):", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    rainfall = st.number_input("🌧️ Rainfall (mm):", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
    district_input = st.text_input("📍 District Name (e.g., ryk):", "ryk")

# Add Farming Practices (for personalization)
st.subheader("🌾 Farming Practices")
irrigation_method = st.selectbox("💧 Irrigation Method:", ["Drip", "Sprinkler", "Flood", "None"])
fertilizer_type = st.selectbox("🌱 Fertilizer Type:", ["Organic", "Chemical", "Mixed", "None"])

# Predict button
if st.button("🚀 Predict Crop"):
    # Prepare input data
    new_data = pd.DataFrame({
        'N': [N],
        'P': [P],
        'K': [K],
        'temperature': [temperature],
        'humidity': [humidity],
        'ph': [ph],
        'rainfall': [rainfall],
        'irrigation_method': [irrigation_method],
        'fertilizer_type': [fertilizer_type]
    })

    # Add district columns (one-hot encoded)
    for col in feature_columns:
        if "district_" in col:
            new_data[col] = 1 if col == f"district_{district_input}" else 0

    # Add missing columns with 0
    for col in feature_columns:
        if col not in new_data.columns:
            new_data[col] = 0

    # Reorder columns to match the training data
    new_data = new_data[feature_columns]

    # Make prediction
    prediction = model.predict(new_data)
    predicted_crop = label_encoder.inverse_transform(prediction)

    # Provide personalized recommendations based on the farming practices
    st.success(f"🌾 **Predicted Crop Type:** {predicted_crop[0]} 🌱")

    # Show visualization (compare this crop with others)
    st.subheader("🌾 Crop Comparison")
    crop_comparison = {
        "Crop Type": ["Crop 1", "Crop 2", "Crop 3"],  # Example placeholder crops
        "Expected Yield (kg/ha)": [3500, 2900, 3000],
        "Water Requirements (mm)": [400, 300, 350]
    }
    comparison_df = pd.DataFrame(crop_comparison)
    st.write(comparison_df)

    # Display farming tips
    st.subheader("🌿 Farming Tips")
    if irrigation_method == "Drip":
        st.write("💧 Drip irrigation is highly efficient, especially for crops requiring moderate water.")
    elif irrigation_method == "Sprinkler":
        st.write("💦 Sprinkler irrigation can cover a large area but may be less water-efficient.")
    else:
        st.write("🌱 For flood irrigation, ensure water is evenly distributed for optimal crop growth.")

    if fertilizer_type == "Organic":
        st.write("🌿 Organic fertilizers improve soil health over time and are great for sustainable farming.")
    elif fertilizer_type == "Chemical":
        st.write("💥 Chemical fertilizers provide fast results but need to be used carefully to avoid soil degradation.")
    else:
        st.write("🌱 Mixed fertilizers can balance the benefits of both organic and chemical options.")
    
# # Footer 
# st.write("**Developed by Us with 💚 for Smart Agriculture 🚜.**")
# Footer with Contact or Additional Information
st.markdown("""
---
**Developed by Us with 💚 for Smart Agriculture 🚜.**  
For more information or support, contact us at [support@agrismart.com](mailto:support@agrismart.com).
""")












