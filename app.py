import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import dlib
import os
from PIL import Image
import matplotlib.font_manager as fm
import gdown

# -----------------------------
# Google Drive Model URL
# -----------------------------
MODEL_URL = "https://drive.google.com/uc?id=1-JNL73G3fSpJZuCPEfTjSdXaXOpHtI6h"
MODEL_LOCAL = "best_inceptionresnetv2_face_shape.keras"

# -----------------------------
# Download model if not exists
# -----------------------------
if not os.path.exists(MODEL_LOCAL):
    gdown.download(MODEL_URL, MODEL_LOCAL, quiet=False)

# -----------------------------
# Caching model load
# -----------------------------
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

face_shape_model = load_model(MODEL_LOCAL)

# -----------------------------
# Dlib setup
# -----------------------------
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
if not os.path.exists(PREDICTOR_PATH):
    import urllib.request
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
    import bz2
    with bz2.BZ2File("shape_predictor_68_face_landmarks.dat.bz2") as fr, open(PREDICTOR_PATH, "wb") as fw:
        fw.write(fr.read())

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# -----------------------------
# Fonts for Thai
# -----------------------------
font_path_sarabun = "/usr/share/fonts/truetype/th-sarabun-new/THSarabunNew.ttf"
font_prop = fm.FontProperties(fname=font_path_sarabun) if os.path.exists(font_path_sarabun) else None

# -----------------------------
# Class labels & hairstyle recommendations
# -----------------------------
classes = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']
hairstyle_recommendations = {
    'Oval': 'ผมสั้นถึงกลาง เช่น blunt bob, shoulder-length, pixie cut, long layers และหน้าม้าปัดข้าง',
    'Square': 'ผมยาวปานกลางถึงยาว พร้อมไล่เลเยอร์หรือปลายฟุ้ง เช่น beach waves และหน้าม้านุ่มๆ',
    'Round': 'ทรงเพิ่มความสูงให้ใบหน้า เช่น textured bob, long layers, แสกข้าง และ blunt bangs',
    'Heart': 'ผมยาวระดับไหล่ พร้อมเลเยอร์บริเวณกราม curtain bangs หรือ wispy bangs',
    'Oblong': 'ลอนคลาย, loose curls, layered bob และหน้าม้าปัดข้างหรือ curtain bangs'
}

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Face Shape Detector", layout="centered")
st.markdown("""
<style>
body {background: linear-gradient(to right, #e0f7fa, #fff9c4);}
h1 {text-align:center; color:#004d40;}
.stButton>button {background-color:#00796b; color:white; border-radius:8px; font-weight:bold;}
.stImage>div>figcaption {text-align:center; font-style:italic; color:#004d40;}
</style>
""", unsafe_allow_html=True)

st.title("Face Shape Detector & Hairstyle Recommendation")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])

SAVE_DIR = "saved_results"
os.makedirs(SAVE_DIR, exist_ok=True)

def predict_face_shape(img_pil):
    img = np.array(img_pil.convert("RGB"))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_landmarks = img_rgb.copy()

    # Resize for model
    img_resized = cv2.resize(img_rgb, (299, 299))
    img_input = np.expand_dims(img_resized, axis=0)
    pred = face_shape_model.predict(img_input, verbose=0)
    idx = np.argmax(pred)
    face_shape = classes[idx]
    confidence = pred[0][idx]*100

    # Landmark detection
    faces = detector(gray)
    ratiog = 0
    score = 0
    if len(faces)>0:
        for face in faces:
            landmarks = predictor(gray, face)
            for i in range(68):
                x = landmarks.part(i).x
                y = landmarks.part(i).y
                cv2.circle(img_landmarks,(x,y),1,(0,255,0),-1)
            chin = landmarks.part(8)
            forehead = landmarks.part(27)
            left_face = landmarks.part(0)
            right_face = landmarks.part(16)
            face_height = abs(chin.y-forehead.y)
            face_width = abs(right_face.x-left_face.x)
            if face_width>0:
                ratiog = face_height/face_width
                score = max(0,min((1-abs(ratiog-1.618)/1.618)*100,100))
    else:
        return "No face detected", None

    recommend = hairstyle_recommendations[face_shape]

    # Save landmark image
    fname = f"{SAVE_DIR}/landmarks_{np.random.randint(0,9999)}.png"
    cv2.imwrite(fname, cv2.cvtColor(img_landmarks, cv2.COLOR_RGB2BGR))

    return f"Face Shape: {face_shape} ({confidence:.2f}%)\nHairstyle: {recommend}\nGolden Ratio: {ratiog:.2f} | Score: {score:.2f}%", img_landmarks

if uploaded_file is not None:
    img_pil = Image.open(uploaded_file)
    result_text, landmark_img = predict_face_shape(img_pil)
    st.text_area("Prediction Result", result_text, height=120)
    st.image(landmark_img, caption="Landmarks Detected", use_column_width=True)
