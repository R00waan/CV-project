import cv2
from PIL import Image
import streamlit as st
from ultralytics import YOLO

st.set_page_config(
    page_title="Construction Site Safety Detection",
    page_icon="🦺",
    layout="wide",
)

st.title("🦺 Construction Site Safety Detection (YOLOv8)")
st.write(
    "Upload an image to detect safety gear, violations, and equipment using"
    " the trained model."
)

st.sidebar.header("Prediction Settings")
conf_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.35,
    step=0.05,
)

MODEL_PATH = "best.pt"


@st.cache_resource
def load_model(path: str):
  return YOLO(path)


try:
  model = load_model(MODEL_PATH)
except Exception as e:
  st.error(
      f"Failed to load model from '{MODEL_PATH}'. Make sure the file is in"
      f" the same folder as app.py.\n\nDetails: {e}"
  )
  st.stop()

uploaded_file = st.file_uploader(
    "Choose an image (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert("RGB")
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Original Image")
    st.image(image, use_container_width=True)

  with st.spinner("Analyzing image..."):
    results = model.predict(source=image, conf=conf_threshold, imgsz=640)
    annotated_bgr = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

  with col2:
    st.subheader("Detection Result")
    st.image(annotated_rgb, use_container_width=True)

  st.subheader("Detected Items Summary")
  boxes = results[0].boxes
  if boxes is not None and len(boxes) > 0:
    summary_data = [
        {
            "Class": model.names[int(cls_id)],
            "Confidence": round(float(conf), 2),
        }
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist())
    ]
    st.dataframe(summary_data, use_container_width=True)
  else:
    st.info(
        "No items detected at the current confidence threshold. Try lowering"
        " it in the sidebar."
    )