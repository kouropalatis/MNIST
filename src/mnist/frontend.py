import streamlit as st
import requests
from PIL import Image
import io

# FastAPI Backend URL
BACKEND_URL = "http://localhost:8000/predict/"

st.set_page_config(
    page_title="MNIST Classifier",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 MNIST Digit Classifier")
st.markdown("Upload a digit image (0-9) to get a prediction from the backend!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Predict button
    if st.button('Predict Digit'):
        with st.spinner('Asking the model...'):
            try:
                # Convert image to bytes for API
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                buf.seek(0)
                
                files = {"file": ("image.png", buf, "image/png")}
                response = requests.post(BACKEND_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Prediction:** {result['prediction']}")
                    st.info(f"**Confidence:** {result['confidence']}")
                    
                    # Optional: Display probabilities
                    if "probabilities" in result:
                        st.expander("See Probabilities").write(result['probabilities'])
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ Could not connect to backend. Is it running on port 8000?")
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

st.markdown("---")
st.caption("Powered by FastAPI & PyTorch")
