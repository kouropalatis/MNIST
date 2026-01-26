import requests
import io
from PIL import Image
import numpy as np

def create_dummy_image():
    # Create a 28x28 random image (simulating MNIST)
    img_data = np.random.randint(0, 255, (28, 28), dtype=np.uint8)
    img = Image.fromarray(img_data)
    
    # Save to buffer
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def test_backend():
    url = "http://127.0.0.1:8000/predict/"
    print(f"Testing {url}...")
    
    try:
        image_buf = create_dummy_image()
        response = requests.post(url, files={"file": ("test_image.png", image_buf, "image/png")})
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", response.json())
            print("✅ Backend verification SUCCESS!")
        else:
            print("Response:", response.text)
            print("❌ Backend verification FAILED (Non-200 response).")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend. Is it running?")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    test_backend()
