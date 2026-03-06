import streamlit as st
import pandas as pd
import joblib

#load our model and encoder
model = joblib.load("course_model.pkl")
encoder = joblib.load("course_encoder.pkl")

st.title("Course Recommendation System")

goal = st.selectbox("Select your goal:",['Job', 'Freelancing', 'Business'])
hobby = st.selectbox("Select your hobby",['Programming','Design','Editing'])

if st.button("Submit"):
    data = pd.DataFrame({
        "goal": [goal],
        "hobby": [hobby]
    })
    data['goal'] = data['goal'].str.lower
    data['hobby'] = data['hobby'].str.lower
    
    encoded_data = encoder.transform(data)
    
    prediction = model.predict(encoded_data)
    predicted = prediction[0]    
    st.success(f"Recommended course: {predicted}")
