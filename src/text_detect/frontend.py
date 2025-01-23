import os
import requests
import streamlit as st
from google.cloud import run_v2

def get_backend_url():
    """Get the URL of the backend service."""
    parent = "projects/my-personal-mlops-project/locations/europe-west1"
    client = run_v2.ServicesClient()
    services = client.list_services(parent=parent)
    for service in services:
        if service.name.split("/")[-1] == "production-model":
            return service.uri
    return os.environ.get("BACKEND", None)

def text_detect(text_data: str, backend: str):
    """
    Send the text data to the backend for classification.
    Expects backend's /predict endpoint to accept JSON:
       { "input_texts": [ <string> ] }
    """
    predict_url = f"{backend}/predict"
    payload = {"input_texts": [text_data]}  # The API expects a list of strings under the key 'input_texts'

    try:
        response = requests.post(predict_url, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Backend error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None


def main() -> None:
    """Main function of the Streamlit frontend."""
    # IMPORTANT: Include the scheme in the URL
    backend = "http://localhost:3000"#get_backend_url()

    st.title("Text Classification Demo")

    uploaded_file = st.file_uploader(
        "Upload a text file", 
        type=["txt"],  # Only .txt files allowed
        help="Upload plain text in UTF-8 format."
    )

    if uploaded_file is not None:
        # We assume it's a .txt file
        raw_file_content = uploaded_file.read()
        # Decode to string
        try:
            text_content = raw_file_content.decode("utf-8", errors="ignore")
        except Exception:
            st.error("Failed to decode file as UTF-8.")
            return

        # Send text_content to the backend
        result = text_detect(text_content, backend=backend)

        if result is not None:
            # If the backend follows the structure:
            #  {
            #    "predicted_classes": [...],
            #    ... or possibly a single string
            #  }
            # For a single input, we can read the first predicted class:
            predictions = result  # Check how your /predict endpoint structures the output
            # Example: If result is something like ["AI", "Human"] for multiple lines,
            #          for a single text input you have just one item:
            if isinstance(predictions, list) and len(predictions) == 1:
                predicted_class = predictions[0]
            else:
                predicted_class = "Unknown"

            st.subheader("Uploaded Text")
            st.text_area("File Content", text_content, height=200)

            st.subheader("Model Prediction")
            st.write(f"**Predicted Class:** {predicted_class}")
        else:
            st.error("No valid response from the backend.")

    else:
        st.info("Please upload a .txt file to get started.")


if __name__ == "__main__":
    main()
