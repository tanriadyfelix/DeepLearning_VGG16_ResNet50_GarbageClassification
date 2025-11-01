import os
from io import BytesIO
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
from huggingface_hub import hf_hub_download

# ============== APP CONFIG ==============
st.set_page_config(
    page_title="Garbage Classification",
    page_icon="♻",
    layout="centered"
)

# === Konfigurasi ===
INPUT_SIZE = (224, 224)

USE_HF = True
HF_REPO_ID = "coconud/DeepLearning_GarbageClassification"
HF_VGG_FILENAME = "garbage_classifier_VGG16.keras"
HF_RESNET_FILENAME = "garbage_classifier_ResNet50.keras"

# === Label & Deskripsi Sampah ===
CLASS_NAMES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes',
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

LABEL_EXPLANATION = {
    "battery": "Sampah baterai termasuk limbah berbahaya (B3). Jangan dibuang sembarangan, serahkan ke tempat daur ulang khusus atau e-waste center.",
    "biological": "Sampah organik seperti sisa makanan atau daun. Dapat dijadikan kompos.",
    "brown-glass": "Sampah kaca coklat (misalnya botol minuman). Daur ulang bisa dilakukan, pisahkan dari kaca warna lain.",
    "cardboard": "Sampah kardus atau karton. Bersihkan dari minyak/air sebelum didaur ulang.",
    "clothes": "Sampah tekstil seperti pakaian bekas. Dapat disumbangkan, digunakan ulang, atau didaur ulang menjadi bahan lain.",
    "green-glass": "Sampah kaca hijau, biasanya botol minuman. Daur ulang secara terpisah dari warna kaca lain.",
    "metal": "Sampah logam seperti kaleng, besi, atau alumunium. Sangat dapat didaur ulang, kumpulkan secara terpisah.",
    "paper": "Sampah kertas seperti koran, majalah, HVS. Daur ulang memungkinkan, hindari yang basah atau berminyak.",
    "plastic": "Sampah plastik (botol, kantong, wadah). Bersihkan sebelum dibuang agar dapat didaur ulang.",
    "shoes": "Sampah sepatu atau alas kaki. Bisa disumbangkan atau digunakan kembali jika masih layak.",
    "trash": "Sampah residu yang tidak bisa didaur ulang (misalnya campuran atau kontaminasi).",
    "white-glass": "Sampah kaca bening seperti botol air mineral. Pisahkan dari kaca warna lain untuk daur ulang."
}

# === Model Loader ===  
@st.cache_resource(show_spinner=True)
def load_vgg16_model():
    downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_VGG_FILENAME)
    model = load_model(downloaded, compile=False)
    return model

@st.cache_resource(show_spinner=True)
def load_resnet50_model():
    downloaded = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_RESNET_FILENAME)
    model = load_model(downloaded, compile=False)
    return model

#  === Process Image Uploaded ===

def preprocess_image(img_pil: Image.Image, model_name: str) -> np.ndarray:
    img = img_pil.convert("RGB").resize(INPUT_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    if model_name == "VGG16":
        arr = vgg_preprocess(arr.copy())
    else:
        arr = res_preprocess(arr.copy())
    return arr

def predict_image(img_pil: Image.Image, model_name: str):
    if model_name == "VGG16":
        model = load_vgg16_model()
    else:
        model = load_resnet50_model()

    x = preprocess_image(img_pil, model_name)
    probs = model.predict(x, verbose=0)[0]
    # Softmax safety
    exps = np.exp(probs - np.max(probs))
    probs = exps / np.sum(exps)
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

def render_confidence_bar(p: float):
    st.write(f"Confidence: {p:.3f}")
    st.progress(min(max(p, 0.0), 1.0))
    if p <= 0.5:
        st.info("Model tidak yakin (≤ 0.5)")
    elif p <= 0.7:
        st.warning("Model cukup yakin (≤ 0.7)")
    else:
        st.success("Model sangat yakin (> 0.7)")

def go_to(page_key: str):
    st.session_state["page"] = page_key


# ============== NAVIGASI SEDERHANA ==============
if "page" not in st.session_state:
    st.session_state["page"] = "landing"

# ============== LANDING PAGE ==============
def page_landing():
    st.title("Garbage Classification")
    st.subheader("Klasifikasi Jenis Sampah dengan Deep Learning (VGG16 & ResNet50)")
    st.write("""
Aplikasi ini mengklasifikasikan gambar sampah menjadi 12 kategori seperti **battery**, **metal**, **plastic**, dan lainnya.  
Teknologi yang digunakan adalah model CNN pretrained VGG16 dan ResNet50 yang telah di-finetune pada dataset Garbage Classification.
    """)
    st.button("Mulai Klasifikasi", on_click=lambda: go_to("classify"))

# ============== HALAMAN: CLASSIFY ==============
def page_classify():
    st.title("Garbage Classification")

    model_name = st.radio("Pilih Model", ["VGG16", "ResNet50"], horizontal=True)
    uploaded = st.file_uploader("Unggah gambar (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        try:
            img = Image.open(BytesIO(uploaded.read()))
        except Exception as e:
            st.error(f"Gagal membaca gambar: {e}")
            return

        preview = img.convert("RGB").resize(INPUT_SIZE)
        st.caption("Preview Uploaded Image: ")
        st.image(preview, width=224)

        with st.spinner("Memproses gambar..."):
            pred_label, pred_prob, probs = predict_image(img, model_name)

        st.markdown("### Hasil Prediksi")
        st.write(f'Hasil Prediksi: **"{pred_label.capitalize()}"**')
        render_confidence_bar(pred_prob)

        st.markdown("### Penjelasan Singkat")
        st.write(LABEL_EXPLANATION.get(pred_label, "Tidak ada penjelasan."))

        st.markdown("### Probabilitas per Kelas")
        table_rows = [{"Label": name, "Prob": float(probs[i])} for i, name in enumerate(CLASS_NAMES)]
        st.dataframe(table_rows, hide_index=True)

    st.divider()
    st.button("Kembali ke Landing Page", on_click=lambda: go_to("landing"))


# ============== RENDER ==============
if st.session_state["page"] == "landing":
    page_landing()
else:
    page_classify()
