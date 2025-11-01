from io import BytesIO
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import streamlit as st
import tensorflow as tf
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as res_preprocess
from huggingface_hub import hf_hub_download

# ==================== APP CONFIG & CUSTOM CSS ====================
st.set_page_config(
    page_title="Garbage Classification",
    page_icon="♻",
    layout="centered"
)

# === Custom CSS untuk Style ===
st.markdown("""
<style>
h1 {
    color: #66BB6A !important;  
    text-align: center; 
}

/* Tombol Upload & Navigasi*/
.stButton>button {
    color: white;
    background-color: #4CAF50; 
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
}

.stButton>button:hover {
    background-color: #66BB6A;
}

/* Teks nama file setelah upload */
.stFileUploader span {
    color: black !important;
}

/* Gambar di tengah */
div.stImage {
    display: flex;
    justify-content: center;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)
# ==================================

# === Konfigurasi ===
INPUT_SIZE = (224, 224)

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

#  === Image Utilities ===

def _letterbox(img: Image.Image, size=(224, 224), fill=(0, 0, 0)):
    """Mengubah ukuran gambar sambil mempertahankan rasio aspek dan menambahkan padding."""
    img = img.convert("RGB")
    ratio = min(size[0] / img.width, size[1] / img.height)
    new_w, new_h = int(img.width * ratio), int(img.height * ratio)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    new_img = Image.new("RGB", size, fill)
    paste_x = (size[0] - new_w) // 2
    paste_y = (size[1] - new_h) // 2
    new_img.paste(img_resized, (paste_x, paste_y))
    return new_img


def _tta_variants(img: Image.Image):
    """Menghasilkan variasi gambar untuk Test Time Augmentation (TTA)."""
    imgs = []
    base = _letterbox(img, INPUT_SIZE)
    imgs.append(base)
    imgs.append(ImageOps.mirror(base))
    imgs.append(ImageOps.flip(base))
    imgs.append(base.rotate(10, resample=Image.BILINEAR))
    imgs.append(base.rotate(-10, resample=Image.BILINEAR))
    imgs.append(base.filter(ImageFilter.SHARPEN))
    return imgs


def _preprocess_arr(img_pil: Image.Image, model_name: str) -> np.ndarray:
    """Konversi PIL image ke numpy array dan pre-process untuk model."""
    arr = np.array(img_pil, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    if model_name == "VGG16":
        arr = vgg_preprocess(arr.copy())
    else:
        arr = res_preprocess(arr.copy())
    return arr


def predict_image(img_pil: Image.Image, model_name: str):
    """Melakukan prediksi menggunakan TTA dan menggabungkan hasilnya."""
    model = load_vgg16_model() if model_name == "VGG16" else load_resnet50_model()

    imgs = _tta_variants(img_pil)
    probs_accum = np.zeros(len(CLASS_NAMES), dtype=np.float32)

    for im in imgs:
        x = _preprocess_arr(im, model_name) 
        preds = model.predict(x, verbose=0)[0]

        probs_accum += preds

    probs = probs_accum / len(imgs)
    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs

def render_confidence_bar(p: float):
    pct = p * 100
    st.write(f"Confidence: **{pct:.2f}%**")
    st.progress(min(max(p, 0.0), 1.0))
    if p <= 0.5:
        st.info("Model tidak yakin (≤ 50%)")
    elif p <= 0.7:
        st.warning("Model cukup yakin (≤ 70%)")
    else:
        st.success("Model sangat yakin (> 70%)")

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
        
        # MENGGUNAKAN KOLOM UNTUK TATA LETAK LEBIH BAIK
        col_img, col_pred = st.columns([1, 2])

        with col_img:
            preview = _letterbox(img, INPUT_SIZE)
            st.caption("Preview Uploaded Image:")
            st.image(preview, width=224)

        with col_pred:
            with st.spinner("Memproses gambar..."):
                pred_label, pred_prob, probs = predict_image(img, model_name)

            # ==================== Hasil Prediksi ====================
            st.markdown("### Hasil Prediksi")
            st.write(f'Hasil Prediksi: **"{pred_label.capitalize()}"**')
            render_confidence_bar(pred_prob)

            # ==================== Penjelasan Singkat ====================
            st.markdown("### Penjelasan Singkat")
            st.write(LABEL_EXPLANATION.get(pred_label, "Tidak ada penjelasan."))


        # ==================== Visualisasi Top-5 ====================
        st.markdown("---")
        st.markdown("### Confidence Level (Top-5)")
        probs_percent = [float(p) * 100 for p in probs]

        probs_vis_df = pd.DataFrame({
            "Label": CLASS_NAMES,
            "Prob": probs_percent
        }).sort_values("Prob", ascending=False).head(5)

        fig = px.bar(
            probs_vis_df,
            x="Prob", y="Label",
            orientation="h",
            text=[f"{p:.2f}%" for p in probs_vis_df["Prob"]],
            color="Prob",
            color_continuous_scale="oryel"
        )
        
        # Peningkatan Plotly UI
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="Label",
            height=400,
            showlegend=False,
            margin=dict(l=60, r=40, t=50, b=40),
            width=None,
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)' 
        )
        st.plotly_chart(fig, width='stretch') 

    st.divider()
    st.button("Kembali ke Landing Page", on_click=lambda: go_to("landing"))


# ============== RENDER APLIKASI ====================
if st.session_state["page"] == "landing":
    page_landing()
else:
    page_classify()
