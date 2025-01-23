import os
import pandas as pd
import requests
import streamlit as st
from google.cloud import run_v2
from google.cloud.run_v2 import ServicesClient


def get_backend_url():
    "Get the url of the backend service"
    parent = "C:/Users/arttu/Desktop/mlops/dtu_22/mlops_project_group_37/src/text_detect/app.py"
    client=run_v2.ServicesClient()
    services=client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1]=='production_model':
            return service.uri
    return os.environ.get("BACKEND", None)

def text_detect(text_file, backend):
    """Send the text to to backend for classification"""
    predict_url = f"{backend}/predict"
    response = requests. post(predict_url, files={"iamge": text_file}, timeout=10)
    if response.status_code==200:
        return response.json()
    return None

def main() -> None:
    """Main fucntion of the Streamlit frontend"""
    backend = get_backend_url()
    if backend is None:
        msg = "Backend service is not found"
        raise ValueError(msg)
    st.title("Text Classification")
    uploaded_file=st.file_uploader("Upload an image", type=["txt", "pdf"])
    
    if uploaded_file is not None:
        text=uploaded_file.read()
        result=text_detect(text, backend=backend)
        
        if result is not None:
            prediction=result["prediciton"]
        "show the text and prediciton"
        st.image(text, caption="Uploaded Text")
        st.write("Prediction:", prediction)
    else:
        st.write("Failed to get prediction")
        
if __name__=="__main__":
    main()
        