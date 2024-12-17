# # Step 1: Import necessary libraries
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import LabelEncoder
# import streamlit as st

# # Streamlit page configuration
# st.set_page_config(
#     page_title="AgriSmart Crop Advisor 🌾",
#     page_icon="🌱",
#     layout="centered",
# )

# # Title and description
# st.title("🌾 AgriSmart Crop Advisor 🌱")
# st.write("""
# Welcome to **AgriSmart Crop Advisor**! This app helps farmers and agricultural consultants predict the best crop to grow 
# based on soil, weather, and location data. 🚜💡
# """)

# # Load and preprocess the dataset
# @st.cache_data
# def load_and_preprocess_data():
#     # Load dataset directly from the script folder
#     file_path = "Crop(Distric level).csv"  # Add your CSV file to the same directory
#     data = pd.read_csv(file_path)
    
#     # Handle missing values
#     data = data.dropna()
    
#     # Encode the target 'label' column
#     label_encoder = LabelEncoder()
#     data['label'] = label_encoder.fit_transform(data['label'])
    
#     # One-hot encode the 'district' column
#     data = pd.get_dummies(data, columns=['district'], drop_first=True)
    
#     # Split features and target
#     X = data.drop('label', axis=1)
#     y = data['label']
    
#     # Split data into training and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Train the model
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
    
#     return model, label_encoder, X.columns

# # Load model and label encoder
# model, label_encoder, feature_columns = load_and_preprocess_data()

# # User Input Interface
# st.subheader("🌟 Enter Input Values for Crop Prediction 🌟")

# # Input fields for user
# col1, col2 = st.columns(2)

# with col1:
#     N = st.number_input("🌱 Nitrogen (N):", min_value=0.0, max_value=200.0, value=50.0, step=1.0)
#     P = st.number_input("🌿 Phosphorus (P):", min_value=0.0, max_value=200.0, value=30.0, step=1.0)
#     K = st.number_input("🍂 Potassium (K):", min_value=0.0, max_value=200.0, value=40.0, step=1.0)
#     ph = st.number_input("🧪 Soil pH:", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

# with col2:
#     temperature = st.number_input("🌡️ Temperature (°C):", min_value=-10.0, max_value=50.0, value=25.0, step=0.1)
#     humidity = st.number_input("💧 Humidity (%):", min_value=0.0, max_value=100.0, value=70.0, step=0.1)
#     rainfall = st.number_input("🌧️ Rainfall (mm):", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
#     district_input = st.text_input("📍 District Name (e.g., ryk):", "ryk")

# # Predict button
# if st.button("🚀 Predict Crop"):
#     # Prepare input data
#     new_data = pd.DataFrame({
#         'N': [N],
#         'P': [P],
#         'K': [K],
#         'temperature': [temperature],
#         'humidity': [humidity],
#         'ph': [ph],
#         'rainfall': [rainfall]
#     })

#     # Add district columns (one-hot encoded)
#     for col in feature_columns:
#         if "district_" in col:
#             new_data[col] = 1 if col == f"district_{district_input}" else 0

#     # Add missing columns with 0
#     for col in feature_columns:
#         if col not in new_data.columns:
#             new_data[col] = 0

#     # Reorder columns to match the training data
#     new_data = new_data[feature_columns]

#     # Make prediction
#     prediction = model.predict(new_data)
#     predicted_crop = label_encoder.inverse_transform(prediction)

#     # Display result
#     st.success(f"🌾 **Predicted Crop Type:** {predicted_crop[0]} 🌱")
#     st.balloons()

# # Footer 
# st.write("**Developed by Us3 with 💚 for Smart Agriculture 🚜.**")
# st.markdown("Stay sustainable, stay productive! 🌍")



import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset directly from the script folder in the GitHub repo
@st.cache_data
def load_and_preprocess_data():
    file_path = "Crop(Distric level).csv"  # Ensure this is in the same directory as your script
    data = pd.read_csv(file_path)
    return data

# Load the dataset
data = load_and_preprocess_data()

# Define feature columns (updated to match your dataset columns)
feature_columns = ['N', 'P', 'K', 'rainfall', 'temperature', 'humidity', 'ph']

# Define target variable
target = 'label'  # Your target variable is 'label' not 'crop_type'

# Model training
X = data[feature_columns]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI setup
st.title("🌾 AgriSmart Crop Advisor")

# Input section
st.sidebar.header("Enter Your District Details")

district_input = st.sidebar.text_input("District Name")
N_input = st.sidebar.number_input("Nitrogen (N) in Soil", min_value=0, max_value=100, value=50)
P_input = st.sidebar.number_input("Phosphorus (P) in Soil", min_value=0, max_value=100, value=50)
K_input = st.sidebar.number_input("Potassium (K) in Soil", min_value=0, max_value=100, value=50)
rainfall_input = st.sidebar.number_input("Rainfall (mm)", min_value=0, max_value=500, value=100)
temperature_input = st.sidebar.number_input("Temperature (°C)", min_value=-10, max_value=50, value=25)
humidity_input = st.sidebar.number_input("Humidity (%)", min_value=0, max_value=100, value=60)
pH_input = st.sidebar.number_input("Soil pH", min_value=4.0, max_value=8.0, value=6.5)

# Weather API Integration (OpenWeather)
api_key = 'your_api_key'  # Replace this with your actual OpenWeather API key
weather_url = f'http://api.openweathermap.org/data/2.5/weather?q={district_input}&appid={api_key}'
weather_response = requests.get(weather_url)
weather_data = weather_response.json()

if 'main' in weather_data:
    temperature = weather_data['main']['temp'] - 273.15  # Convert from Kelvin to Celsius
    st.write(f"🌡️ Current Temperature in {district_input}: {temperature:.2f}°C")
else:
    st.write("Weather data not available.")

# Prepare input data for prediction
new_data = pd.DataFrame({
    'N': [N_input],
    'P': [P_input],
    'K': [K_input],
    'rainfall': [rainfall_input],
    'temperature': [temperature_input],
    'humidity': [humidity_input],
    'ph': [pH_input]
})

# Make prediction
predicted_crop = model.predict(new_data)

# Confidence score
prediction_prob = model.predict_proba(new_data)
confidence = max(prediction_prob[0]) * 100
st.write(f"🟢 Confidence Level: {confidence:.2f}%")

# Crop Details
crop_details = {
    "Wheat": "Wheat grows best in loamy soil with good drainage. Requires moderate rainfall and temperature around 15-20°C.",
    "Rice": "Rice needs flooded fields and warm climates. It requires temperatures around 20-35°C.",
    # Add more crops and their details here
}

st.write(f"🌾 **Predicted Crop**: {predicted_crop[0]}")
st.write(f"🌿 **Crop Information**: {crop_details.get(predicted_crop[0], 'No detailed info available.')}")
    
# Feature Importance Visualization
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': feature_importances
})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

st.subheader("📊 Feature Importance")
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
st.pyplot(fig)

# User Feedback Loop
feedback = st.radio("Was the crop prediction helpful?", ('Yes', 'No'))
if feedback == 'No':
    st.text_area("Please provide feedback to improve the app:")

# Crop Recommendations Based on Weather
if temperature > 30:
    st.write("☀️ Consider crops that thrive in hot climates like Rice or Sorghum.")
else:
    st.write("🌱 Consider cool-weather crops like Wheat or Barley.")

# Offline Mode (simulate saving predictions locally or sending reports)
st.sidebar.header("Offline Features")
offline_option = st.sidebar.radio("Save your prediction?", ('Yes', 'No'))
if offline_option == 'Yes':
    st.write("Your prediction will be saved for offline use.")

# Mobile-Optimized UI
st.sidebar.text("For better mobile experience, please make sure the app is optimized with larger buttons and simpler input fields.")

# District-Level Collaboration and Localization
st.sidebar.header("Collaborate with Local Farmers")
st.sidebar.text("Share your experiences with others in your district and exchange tips.")

# Final Remarks
st.write("Thank you for using **AgriSmart Crop Advisor**! Stay connected for more features.")
