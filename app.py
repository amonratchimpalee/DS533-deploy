
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import dlib
import os
import random
import matplotlib.font_manager as fm

# Import the correct preprocess_input for InceptionResNetV2
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input as inceptionresnet_preprocess_input

# -----------------------------
# ฟังก์ชัน preprocess (Preprocess function)
# -----------------------------
def preprocess_for_inceptionresnet(x):
    x = tf.cast(x, tf.float32)
    return inceptionresnet_preprocess_input(x)

# -----------------------------
# โหลดฟอนต์ภาษาไทย (Load Thai fonts)
# -----------------------------
# This block should only be run once for setup, or handled outside Streamlit's main loop
# For simplicity in this app.py, we'll try to ensure it's available.
# In a real Streamlit deployment, these would typically be pre-installed or handled in setup.

# Suppress output from apt-get commands
import subprocess

def install_thai_fonts():
    try:
        # Ensure non-interactive mode for apt-get to prevent hanging
        subprocess.run(['apt-get', 'update', '-qq'], check=True, capture_output=True, text=True)
        subprocess.run(['apt-get', 'install', '-y', 'fonts-thai-tlwg'], check=True, capture_output=True, text=True)
        print("Thai fonts installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing Thai fonts: {e.stderr}")
    except FileNotFoundError:
        print("apt-get command not found. Skipping font installation.")

install_thai_fonts()

font_path_sarabun = "/usr/share/fonts/truetype/th-sarabun-new/THSarabunNew.ttf"
font_path_loma = "/usr/share/fonts/truetype/tlwg/Loma.ttf"

font_prop = None
if os.path.exists(font_path_sarabun):
    font_prop = fm.FontProperties(fname=font_path_sarabun)
    # st.write(f"Using Thai font: {font_path_sarabun}") # Streamlit doesn't show print/write during init
elif os.path.exists(font_path_loma):
    font_prop = fm.FontProperties(fname=font_path_loma)
    # st.write(f"Using Thai font: {font_path_loma}")
else:
    try:
        generic_sans_serif_path = fm.findfont(fm.FontProperties(family='sans-serif'))
        font_prop = fm.FontProperties(fname=generic_sans_serif_path)
        # st.write(f"Falling back to system sans-serif: {generic_sans_serif_path}.")
    except Exception:
        font_prop = fm.FontProperties() # Final fallback

# -----------------------------
# โหลดโมเดล Face Shape (Load Face Shape Model)
# -----------------------------
@st.cache_resource
def load_face_shape_model():
    model = tf.keras.models.load_model(
        '/content/drive/MyDrive/best_inceptionresnetv2_face_shape.keras',
        safe_mode=False,
        custom_objects={'preprocess': preprocess_for_inceptionresnet}
    )
    return model

face_shape_model = load_face_shape_model()

# -----------------------------
# โหลด dlib landmark model (Load dlib landmark model)
# -----------------------------
@st.cache_resource
def load_dlib_models():
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        # Use subprocess to run shell commands to ensure Streamlit can execute them
        try:
            st.info("Downloading shape_predictor_68_face_landmarks.dat...")
            # Capture output to avoid polluting Streamlit logs with wget/bzip2 messages
            subprocess.run(['wget', '-q', 'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2'], check=True, capture_output=True, text=True)
            subprocess.run(['bzip2', '-dk', 'shape_predictor_68_face_landmarks.dat.bz2'], check=True, capture_output=True, text=True)
            st.success("Shape predictor downloaded successfully.")
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to download shape_predictor_68_face_landmarks.dat: {e.stderr}")
            st.stop()
        except FileNotFoundError:
            st.error("wget or bzip2 command not found. Cannot download dlib model.")
            st.stop()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    return detector, predictor

detector, predictor = load_dlib_models()

# -----------------------------
# คลาสรูปหน้า (Face Shape Classes)
# -----------------------------
classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# -----------------------------
# Hairstyle Recommendation (อ้างอิง Paper)
# -----------------------------
hairstyle_recommendations = {
    'Oval': 'ผมสั้นถึงกลาง เช่น blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง ช่วยเน้นดวงตาและโหนกแก้ม',
    'Square': 'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง เช่น beach waves และหน้าม้านุ่มๆ ช่วยลดความเหลี่ยมของกราม',
    'Round': 'ทรงเพิ่มความสูงให้ใบหน้า เช่น textured bob, long layers, แสกข้าง และ blunt bangs ช่วยให้หน้าดูยาวขึ้น',
    'Heart': 'ผมยาวระดับไหล่ พร้อมเลเยอร์บริเวณกราม curtain bangs หรือ wispy bangs ช่วยบาลานซ์หน้าผากกว้าง',
    'Oblong': 'ลอนคลาย, loose curls, layered bob และหน้าม้าปัดข้างหรือ curtain bangs ช่วยเพิ่มความกว้างให้ใบหน้า'
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(layout="wide")
st.title("Face Shape Detector & Hairstyle Recommendation")
st.write("Upload an image to detect face shape, landmarks, golden ratio and get hairstyle recommendation.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image from Streamlit
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Cannot decode image. Please upload a valid image file.")
        st.stop()

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_landmarks = img_rgb.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # ทำนาย face shape (Predict Face Shape)
    # -----------------------------
    IMG_SIZE_MODEL = 299 # Assuming the model expects 299x299 as per InceptionResNetV2
    img_resized = cv2.resize(img, (IMG_SIZE_MODEL, IMG_SIZE_MODEL))
    img_processed = preprocess_for_inceptionresnet(img_resized)
    img_input = np.expand_dims(img_processed, axis=0)

    pred = face_shape_model.predict(img_input, verbose=0)
    idx = np.argmax(pred)
    face_shape = classes[idx]
    confidence = pred[0][idx] * 100

    # -----------------------------
    # ตรวจจับ landmark และ golden ratio (Detect landmark and golden ratio)
    # -----------------------------
    faces = detector(gray)
    ratiog = 0.0
    score = 0.0

    if len(faces) == 0:
        st.warning("No face detected in the image.")
        display_face_shape = "N/A"
        display_recommend = "No recommendation (face not detected)"
    else:
        for face in faces:
            landmarks = predictor(gray, face)
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(img_landmarks, (x, y), 1, (0, 255, 0), -1)

            chin = landmarks.part(8)
            forehead = landmarks.part(27)
            left_face = landmarks.part(0)
            right_face = landmarks.part(16)

            face_height = abs(chin.y - forehead.y)
            face_width = abs(right_face.x - left_face.x)
            golden_ratio = 1.618
            if face_width > 0:
                ratiog = face_height / face_width
                score = (1 - abs(ratiog - golden_ratio) / golden_ratio) * 100
                score = max(0, min(score, 100))

        display_face_shape = face_shape
        display_recommend = hairstyle_recommendations.get(face_shape, "No specific recommendation available.")

    # -----------------------------
    # แสดงผล (Display results)
    # -----------------------------
    st.subheader("Result")
    st.markdown(f"**Face Shape**: {display_face_shape} ({confidence:.2f}%) (confidence for predicted shape)")

    # Use st.markdown with proper font if needed, otherwise Streamlit's default font is fine
  if font_prop and font_prop.get_file():  # ตรวจสอบว่าโหลดฟอนต์ได้
    st.markdown(
        f'<div style="font-family:\'{font_prop.get_name()}\'; font-size:18px;">'
        f'<b>ทรงผมแนะนำ</b>: {display_recommend}</div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(f"**Hairstyle Recommendation**: {display_recommend}")

    st.write(f"Golden Ratio: {ratiog:.2f} | Score: {score:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Original Image", use_column_width=True)
    with col2:
        st.image(img_landmarks, caption="Landmarks Detected", use_column_width=True)
