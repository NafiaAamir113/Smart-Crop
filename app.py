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

# Assuming 'data' is the DataFrame you are using for your crop recommendations
data = pd.DataFrame({
    'N': [90], 
    'P': [42], 
    'K': [43], 
    'temperature': [20.88], 
    'humidity': [82.00], 
    'ph': [6.50], 
    'rainfall': [202.94], 
    'label': ['rice'],
    'district': ['ryk']
})

# If you're using a user input for selecting a district and feature columns, this would look like:
district_input = st.sidebar.text_input("District Name")

# Example logic to filter the data based on district
filtered_data = data[data['district'] == district_input]

# If data for the selected district exists
if not filtered_data.empty:
    st.write(f"Data for district: {district_input}")
    st.write(filtered_data)
    
    # Use the features you want for prediction or recommendation (example: N, P, K, etc.)
    feature_columns = ['N', 'P', 'K', 'humidity', 'ph', 'rainfall']  # Define features
    X = filtered_data[feature_columns]  # Get features for the selected district
    
    # Sample condition to recommend crops based on N, P, K values (example logic)
    if X['N'].iloc[0] > 50 and X['P'].iloc[0] > 30:
        st.write("Recommend crop: Rice")
    else:
        st.write("Recommend crop: Wheat")
else:
    st.write("No data found for this district. Please try a different district.")

# Continue with the rest of your logic for the app

