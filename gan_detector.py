import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2  
import io

# 🎨 Apply Custom Light Theme Styling
st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF;
        color: #000000;
        font-family: Arial, sans-serif;
    }
    .stApp {
        background-color: #FFFFFF;
        padding: 20px;
    }
    .stFileUploader, .stButton>button {
        background-color: #F0F0F0 !important;
        color: #000000 !important;
        font-size: 16px;
        border-radius: 10px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #FFD700 !important;
        color: #000000 !important;
    }
    .stAlert, .stSuccess, .stError, .stWarning {
        border-radius: 10px;
    }
    .result-box {
        background-color: #F8F8F8;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 🔹 Load Model Function
@st.cache_resource
def load_model():
    model_path = r"C:\Users\LENOVO\OneDrive\Desktop\mini project\mobilenet_best (3).pth"
    try:
        model = models.mobilenet_v3_large(pretrained=False)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 1280),
            nn.Hardswish(),
            nn.Dropout(0.5),
            nn.Linear(1280, 2)
        )
        state_dict = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# 🔹 Image Preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    ])
    return transform(image).unsqueeze(0)

# 🔹 Face Detection Function
def detect_face(image):
    open_cv_image = np.array(image.convert('RGB'))
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2,  
        minNeighbors=5,  
        minSize=(40, 40)  
    )
    return len(faces) > 0  

# 🔹 Prediction Function
def predict(image, model):
    image_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)

    gan_score = probabilities[0, 0].item()  
    real_score = probabilities[0, 1].item()  

    predicted_class = "Real" if real_score > gan_score else "GAN-Generated"
    return predicted_class, real_score, gan_score

# 🎨 Sidebar - Developer Info
st.sidebar.markdown("### 👨‍💻 Developed By")

team_members = [
    {"name": "Ammu Elizabeth Alexander", "image": r"C:\Users\LENOVO\Downloads\WhatsApp Image 2025-08-08 at 9.46.20 PM.jpeg"},
    {"name": "Anakha Prakash", "image": r"C:\Users\LENOVO\Downloads\WhatsApp Image 2025-08-08 at 9.43.02 PM.jpeg"},
    {"name": "Aiswarya Josy", "image": r"C:\Users\LENOVO\Downloads\WhatsApp Image 2025-08-08 at 9.43.02 PM (1).jpeg"},
    {"name": "Abin Joseph", "image": r"C:\Users\LENOVO\Downloads\WhatsApp Image 2025-08-08 at 9.43.01 PM.jpeg"},
]

# Load images safely
def load_local_image(image_path):
    try:
        with open(image_path, "rb") as f:
            return Image.open(io.BytesIO(f.read()))
    except Exception as e:
        return None

for member in team_members:
    image = load_local_image(member["image"])
    col1, col2 = st.sidebar.columns([1, 3])  # Adjust column width for image and name
    with col1:
        if image:
            st.image(image, width=50)
        else:
            st.warning(f"⚠️ {member['name']}'s image not found!")
    with col2:
        st.sidebar.write(f"**{member['name']}**")

st.sidebar.markdown("---")
st.sidebar.markdown("### About our model")

st.sidebar.info("This AI model detects if an image is GAN-generated or real. Simply upload an image and get instant results!")

# 🎯 Streamlit UI
st.markdown("<h1 style='text-align: center; color: #FFD700;'>🔍 GAN Image Detector</h1>", unsafe_allow_html=True)
st.write("Upload an image to analyze whether it's AI-generated or real.")

# Load Model
model = load_model()

# File Upload
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "png", "jpeg"], key="fileUploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    # ✅ Show uploaded image
    st.image(image, caption="📷 Uploaded Image", use_container_width=True, output_format="auto")

    with st.spinner("🔍 Analyzing image..."):
        face_detected = detect_face(image)

    if not face_detected:
        st.warning("⚠️ No face detected. Try another image.")
    else:
        st.success("✅ Face detected!")

        if model:
            st.write("⏳ Processing...")

            # ⏳ Animated Progress Bar
            progress_bar = st.progress(0)  
            for i in range(100):  
                progress_bar.progress(i + 1)

            prediction, real_score, gan_score = predict(image, model)
            
            # 🌟 Stylish result display
            st.markdown(f"""
                <div class='result-box'>
                    <h2 style='color: #FFD700;'>🔍 Prediction: {prediction}</h2>
                    <p><b>📊 Probability Scores</b></p>
                    <p>🟢 <b>Real:</b> {real_score:.4f} | 🔴 <b>GAN:</b> {gan_score:.4f}</p>
                </div>
            """, unsafe_allow_html=True)

            if prediction == "Real":
                st.success("🟢 This image is likely **REAL**! ✅")
            else:
                st.error("🔴 This image is likely **GAN-Generated**! ❌")
        else:
            st.error("⚠️ Model not loaded. Check your file path and restart.")
