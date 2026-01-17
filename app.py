import gradio as gr
import numpy as np
import pandas as pd
import pickle 

# Load Model
with open('health_insurance_fraud_claims_prediction_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
# Main Logic
def predict_claims(ClaimAmount, PatientAge, PatientGender, ProviderSpecialty, ClaimStatus, PatientIncome, PatientMaritalStatus, PatientEmploymentStatus, ClaimType, ClaimSubmissionMethod, Cluster):
    input_data = pd.DataFrame({
        'ClaimAmount': [ClaimAmount],
        'PatientAge': [PatientAge],
        'PatientGender': [PatientGender],
        'ProviderSpecialty': [ProviderSpecialty],
        'ClaimStatus': [ClaimStatus],
        'PatientIncome': [PatientIncome],
        'PatientMaritalStatus': [PatientMaritalStatus],
        'PatientEmploymentStatus': [PatientEmploymentStatus],
        'ClaimType': [ClaimType],
        'ClaimSubmissionMethod': [ClaimSubmissionMethod],
        'Cluster': [Cluster]
    })
    
    # Predict
    prediction = model.predict(input_data)[0]
    
    return "Fraudulent Claim" if prediction == 1 else "Legitimate Claim"

    
inputs = [
    gr.Number(label="Claim Amount"),
    gr.Number(label="Patient Age"),
    gr.Radio(choices=["Male", "Female"], label="Patient"),
    gr.Dropdown(choices=["Cardiology", "Orthopedics", "Neurology", "Pediatrics", "General Surgery"], label="Provider Specialty"),
    gr.Radio(choices=["Approved", "Denied", "Pending"], label="Claim Status"),
    gr.Number(label="Patient Income"),
    gr.Dropdown(choices=["Single", "Married", "Divorced", "Widowed"], label="Patient Marital Status"),
    gr.Dropdown(choices=["Employed", "Unemployed", "Self-Employed", "Retired"], label="Patient Employment Status"),
    gr.Radio(choices=["Inpatient", "Outpatient", "Emergency"], label="Claim Type"),
    gr.Radio(choices=["Online", "Mail", "In-Person"], label="Claim Submission Method"),
    gr.Number(label="Cluster")
]

# Interface
app = gr.Interface(
    fn=predict_claims,
    inputs=inputs,
    outputs="text",
    title="Health Insurance Fraud Claims Prediction",
    description="Predict whether a health insurance claim is fraudulent or legitimate based on various features."
)
app.launch(share=True)