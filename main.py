# import streamlit as st
from ayurvedic_cures import ayurvedic_cures
from model_train import disease_predictor
from symptoms import symptoms
import numpy as np
import pandas as pd 
from urllib.parse import urlencode
from disease import disease
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from urllib.parse import urlencode


data = {'disease':''}

app = FastAPI()

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict_disease(symptoms: dict):
    try:
        predicted_disease = disease_predictor(
            symptoms.get("symptom1", ""),
            symptoms.get("symptom2", ""),
            symptoms.get("symptom3", ""),
            symptoms.get("symptom4", ""),
            symptoms.get("symptom5", ""),
        )
        data['disease'] = predicted_disease

        ayurvedic_curess = ayurvedic_cures.get(data['disease'])
        result = {'disease': data['disease'], 'ayurvedic_cure': ayurvedic_curess}

        return result
        # return predict_disease

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# def mainFunc():
#     st.title("Disease Prediction App")

#     symptom1 = st.selectbox("Symptom 1", ["Select a Symptom"] + symptoms)
#     symptom2 = st.selectbox("Symptom 2", ["Select a Symptom"] + symptoms)
#     symptom3 = st.selectbox("Symptom 3", ["Select a Symptom"] + symptoms)
#     symptom4 = st.selectbox("Symptom 4", ["Select a Symptom"] + symptoms)
#     symptom5 = st.selectbox("Symptom 5", ["Select a Symptom"] + symptoms)

#     if st.button("Submit"):      
#         print(f"printed:{symptom1}")
#         predicted_disease = disease_predictor(symptom1, symptom2, symptom3, symptom4, symptom5)
#         data['disease'] = predicted_disease

    
#     st.write(f"Predicted Disease: {data['disease']}")
#     st.write(f'Ayurvedic Cure:')
#     l = ayurvedic_cures.get(data['disease'], 'No result yet')
#     for i in l:
#         i_parts = i.split(":", 1)
#         searchUrl=urlencode({"q":{i_parts[0]}})
#         st.markdown(f"{i}(https://www.google.com/search?q={searchUrl})")

# if __name__ == "__main__":  
#     mainFunc()
