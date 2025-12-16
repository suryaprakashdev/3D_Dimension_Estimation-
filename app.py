#!/usr/bin/env python3
import os

# ────────────────────────────────────────────────────────────
# 1) FORCE HOME into /tmp so that ~/.streamlit is writable
# ────────────────────────────────────────────────────────────
os.environ["HOME"] = "/tmp"
streamlit_config = os.path.join(os.environ["HOME"], ".streamlit")
os.makedirs(streamlit_config, exist_ok=True)
os.environ["STREAMLIT_CONFIG_DIR"] = streamlit_config

# (Optional) also move Matplotlib & Torch caches into /tmp
os.environ["MPLCONFIGDIR"]   = os.path.join(os.environ["HOME"], ".matplotlib")
os.environ["XDG_CACHE_HOME"] = os.path.join(os.environ["HOME"], ".cache")

# 2) Prepare your own output directory under /tmp
OUTPUT_DIR = os.path.join(os.environ["HOME"], "streamlit_d2_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────
# 3) IMPORT rest of your dependencies
# ────────────────────────────────────────────────────────────
import streamlit as st
import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image, UnidentifiedImageError
import subprocess  # For Detectron2 installation check
import sys         # For Detectron2 installation check

# Detectron2 
d2_imported_successfully = False
try:
    import detectron2
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2.structures import Boxes  # For Bounding Boxes
    d2_imported_successfully = True
    print("Detectron2 utilities imported successfully.")
except ImportError:
    st.error("Detectron2 not found or not installed correctly. Please ensure it's installed in your environment.")
    print("Failed to import Detectron2 utilities.")
except Exception as e:
    st.error(f"An error occurred during Detectron2 imports: {e}")
    print(f"An error occurred during Detectron2 imports: {e}")

# PyTorch
from torchvision import models as torchvision_models
import torch.nn as nn

# Configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CNN_INPUT_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
MODEL_PATH = "model/pix3d_dimension_estimator_mask_crop.pth"

# Dimension Estimation CNN
def create_dimension_estimator_cnn_for_inference(num_outputs=4):
    model = torchvision_models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_outputs)
    )
    return model

@st.cache_resource
def load_dimension_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Dimension estimation model not found at {MODEL_PATH}. Please check the path.")
        return None
    try:
        model = create_dimension_estimator_cnn_for_inference()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print(f"Dimension estimation model loaded from {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"Error loading dimension estimation model: {e}")
        return None

@st.cache_resource
def load_detectron2_model():
    if not d2_imported_successfully:
        return None, None
    try:
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        predictor = DefaultPredictor(cfg)
        print("Detectron2 predictor created.")
        return predictor, cfg
    except Exception as e:
        st.error(f"Error loading Detectron2 model: {e}")
        return None, None

def get_largest_instance_index(instances):
    if not len(instances):
        return -1
    if instances.has("pred_masks"):
        areas = instances.pred_masks.sum(dim=(1,2))
        return int(areas.argmax()) if len(areas) > 0 else 0
    elif instances.has("pred_boxes"):
        boxes = instances.pred_boxes.tensor
        areas = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
        return int(areas.argmax()) if len(areas) > 0 else 0
    return 0

def crop_from_mask(image_np_rgb, mask_tensor):
    mask_np = mask_tensor.cpu().numpy().astype(np.uint8)
    if mask_np.sum() == 0: return None
    rows = np.any(mask_np, axis=1)
    cols = np.any(mask_np, axis=0)
    if not rows.any() or not cols.any(): return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    pad = 5
    ymin = max(0, ymin-pad); xmin = max(0, xmin-pad)
    ymax = min(image_np_rgb.shape[0]-1, ymax+pad)
    xmax = min(image_np_rgb.shape[1]-1, xmax+pad)
    if ymin>=ymax or xmin>=xmax: return None
    return image_np_rgb[ymin:ymax+1, xmin:xmax+1]

def predict_dimensions_cnn(img_rgb, model):
    if model is None:
        return {"Length": "N/A", "Width": "N/A", "Height": "N/A", "Volume": "N/A"}
    try:
        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((CNN_INPUT_SIZE, CNN_INPUT_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])
        inp = transform(img_rgb).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(inp).squeeze().cpu().tolist()
        while len(out)<4: out.append(0.0)
        L, W, H, V = out
        return {
            "Length (cm)": f"{L*100:.1f}",
            "Width (cm)" : f"{W*100:.1f}",
            "Height (cm)": f"{H*100:.1f}",
            "Volume (cm³)": f"{V*1e6:.1f}"
        }
    except Exception as e:
        print(f"CNN predict error: {e}")
        return {"Length": "Error", "Width":"Error", "Height":"Error", "Volume":"Error"}

# ────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Object Dimension Estimator")
st.title("Object Dimension & Volume Estimation")

dim_model = load_dimension_model()
d2_predictor, d2_cfg = (None, None)
d2_metadata = None
if d2_imported_successfully:
    d2_predictor, d2_cfg = load_detectron2_model()
    if d2_cfg:
        try:
            d2_metadata = MetadataCatalog.get(d2_cfg.DATASETS.TRAIN[0])
        except:
            d2_metadata = MetadataCatalog.get("coco_2017_val")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded:
    st.subheader(uploaded.name)
    try:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    except:
        st.error("Invalid image.")
        bgr = None

    if bgr is not None and d2_predictor and dim_model:
        with st.spinner("Processing..."):
            outs = d2_predictor(bgr); inst = outs["instances"].to("cpu")
            if len(inst)==0:
                st.warning("No objects detected.")
            else:
                viz = Visualizer(bgr[:,:,::-1], metadata=d2_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
                out_vis = viz.draw_instance_predictions(inst)
                det_img = out_vis.get_image()[:,:,::-1]
                st.image(det_img, use_column_width=True)

                idx = get_largest_instance_index(inst)
                if idx>=0:
                    mask = inst[idx].pred_masks[0] if inst.has("pred_masks") else None
                    crop = crop_from_mask(img_np, mask) if mask is not None else None
                    if crop is not None:
                        st.image(crop, caption="Cropped Object", width=250)
                        dims = predict_dimensions_cnn(crop, dim_model)
                        st.json(dims)
                    else:
                        st.error("Could not crop object.")
    elif not d2_imported_successfully:
        st.error("Detectron2 not loaded.")
    else:
        st.error("Model not loaded.")

# Sidebar status
st.sidebar.markdown("---")
st.sidebar.write(f"Device: {DEVICE}")
st.sidebar.write(f"Detectron2: {'OK' if d2_predictor else 'Failed'}")
st.sidebar.write(f"Dim CNN: {'OK' if dim_model else 'Failed'}")
