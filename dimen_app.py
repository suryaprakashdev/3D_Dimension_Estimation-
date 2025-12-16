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
from PIL import Image, ExifTags
import subprocess
import sys
import json
from datetime import datetime

# Detectron2 
d2_imported_successfully = False
try:
    import detectron2
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.utils.visualizer import Visualizer, ColorMode
    from detectron2.data import MetadataCatalog
    from detectron2.structures import Boxes
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

# ────────────────────────────────────────────────────────────
# NEW: Camera Feature Extraction
# ────────────────────────────────────────────────────────────
def extract_camera_features(image_file):
    """Extract EXIF metadata and convert to numerical features"""
    try:
        img = Image.open(image_file)
        exif_data = {}
        
        if hasattr(img, '_getexif') and img._getexif() is not None:
            exif = img._getexif()
            for tag_id, value in exif.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                exif_data[tag] = value
        
        # Extract and normalize camera features
        features = {}
        
        # Focal Length (normalize to 0-1 range, typical range 4-200mm)
        focal_length = exif_data.get('FocalLength', None)
        if focal_length:
            if isinstance(focal_length, tuple):
                focal_length = focal_length[0] / focal_length[1] if focal_length[1] != 0 else 0
            features['focal_length'] = min(float(focal_length) / 200.0, 1.0)
        else:
            features['focal_length'] = 0.0
        
        # F-Number/Aperture (normalize, typical range f/1.4 - f/22)
        f_number = exif_data.get('FNumber', None)
        if f_number:
            if isinstance(f_number, tuple):
                f_number = f_number[0] / f_number[1] if f_number[1] != 0 else 0
            features['aperture'] = min(float(f_number) / 22.0, 1.0)
        else:
            features['aperture'] = 0.0
        
        # ISO (normalize, typical range 100-6400)
        iso = exif_data.get('ISOSpeedRatings', None)
        if iso:
            features['iso'] = min(float(iso) / 6400.0, 1.0)
        else:
            features['iso'] = 0.0
        
        # Exposure Time (normalize, typical range 1/8000 - 30s)
        exposure_time = exif_data.get('ExposureTime', None)
        if exposure_time:
            if isinstance(exposure_time, tuple):
                exposure_time = exposure_time[0] / exposure_time[1] if exposure_time[1] != 0 else 0
            # Log scale for exposure time
            features['exposure_time'] = min(np.log10(float(exposure_time) + 1e-6) / 2.0 + 0.5, 1.0)
        else:
            features['exposure_time'] = 0.0
        
        # Image dimensions (normalized by max common resolution)
        features['width'] = min(img.size[0] / 4000.0, 1.0)
        features['height'] = min(img.size[1] / 4000.0, 1.0)
        features['aspect_ratio'] = img.size[0] / img.size[1] if img.size[1] != 0 else 1.0
        
        # Flash (binary: 0 or 1)
        flash = exif_data.get('Flash', 0)
        features['flash'] = 1.0 if flash and int(flash) & 1 else 0.0
        
        # White Balance Mode (0=auto, 1=manual, normalized)
        white_balance = exif_data.get('WhiteBalance', 0)
        features['white_balance'] = float(white_balance) if white_balance else 0.0
        
        # Exposure Mode (normalize 0-2)
        exposure_mode = exif_data.get('ExposureMode', 0)
        features['exposure_mode'] = min(float(exposure_mode) / 2.0, 1.0) if exposure_mode else 0.0
        
        # Store raw metadata for display
        raw_metadata = {
            'make': exif_data.get('Make', 'N/A'),
            'model': exif_data.get('Model', 'N/A'),
            'datetime': exif_data.get('DateTime', 'N/A'),
            'focal_length_mm': f"{focal_length:.1f}mm" if focal_length else 'N/A',
            'aperture_f': f"f/{f_number:.1f}" if f_number else 'N/A',
            'iso': iso if iso else 'N/A',
            'exposure_time_s': f"{exposure_time:.4f}s" if exposure_time else 'N/A',
            'flash': 'Yes' if features['flash'] else 'No',
            'width': img.size[0],
            'height': img.size[1],
        }
        
        # Convert to tensor (10 features)
        feature_vector = torch.tensor([
            features['focal_length'],
            features['aperture'],
            features['iso'],
            features['exposure_time'],
            features['width'],
            features['height'],
            features['aspect_ratio'],
            features['flash'],
            features['white_balance'],
            features['exposure_mode']
        ], dtype=torch.float32)
        
        return feature_vector, raw_metadata
        
    except Exception as e:
        print(f"Error extracting camera features: {e}")
        # Return zero vector if extraction fails
        return torch.zeros(10, dtype=torch.float32), {'error': str(e)}

# ────────────────────────────────────────────────────────────
# MODIFIED: Dimension Estimation CNN with Camera Features
# ────────────────────────────────────────────────────────────
def create_dimension_estimator_cnn_for_inference(num_outputs=4, use_camera_features=True):
    """
    Modified ResNet50 that accepts both image and camera features
    """
    model = torchvision_models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    
    if use_camera_features:
        # Combine ResNet features with camera features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs + 10, 512),  # +10 for camera features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_outputs)
        )
    else:
        # Original architecture (fallback)
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_outputs)
        )
    
    return model

class DimensionEstimatorWithFeatures(nn.Module):
    """
    Wrapper model that combines image CNN with camera features
    """
    def __init__(self, base_model, use_camera_features=True):
        super().__init__()
        self.use_camera_features = use_camera_features
        self.base_model = base_model
        
    def forward(self, img_tensor, camera_features=None):
        if self.use_camera_features and camera_features is not None:
            # Extract features from ResNet (before final FC layer)
            x = self.base_model.conv1(img_tensor)
            x = self.base_model.bn1(x)
            x = self.base_model.relu(x)
            x = self.base_model.maxpool(x)
            x = self.base_model.layer1(x)
            x = self.base_model.layer2(x)
            x = self.base_model.layer3(x)
            x = self.base_model.layer4(x)
            x = self.base_model.avgpool(x)
            x = torch.flatten(x, 1)
            
            # Concatenate with camera features
            camera_features = camera_features.to(x.device)
            combined = torch.cat([x, camera_features], dim=1)
            
            # Pass through final FC layers
            return self.base_model.fc(combined)
        else:
            return self.base_model(img_tensor)

@st.cache_resource
def load_dimension_model(use_camera_features=True):
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Dimension estimation model not found at {MODEL_PATH}. Using base model.")
        model = create_dimension_estimator_cnn_for_inference(use_camera_features=False)
        model.to(DEVICE)
        model.eval()
        return DimensionEstimatorWithFeatures(model, use_camera_features=False), False
    
    try:
        # Try loading with camera features first
        model = create_dimension_estimator_cnn_for_inference(use_camera_features=use_camera_features)
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            print(f"Dimension estimation model loaded from {MODEL_PATH} (with camera features)")
            return DimensionEstimatorWithFeatures(model, use_camera_features=True), True
        except:
            # Fallback to original architecture
            model = create_dimension_estimator_cnn_for_inference(use_camera_features=False)
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
            model.to(DEVICE)
            model.eval()
            print(f"Dimension estimation model loaded from {MODEL_PATH} (original architecture)")
            return DimensionEstimatorWithFeatures(model, use_camera_features=False), False
    except Exception as e:
        st.error(f"Error loading dimension estimation model: {e}")
        return None, False

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

def predict_dimensions_cnn(img_rgb, model, camera_features=None, use_features=False):
    """
    Modified prediction function that accepts camera features
    """
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
        
        # Prepare camera features if available
        cam_feat = None
        if use_features and camera_features is not None:
            cam_feat = camera_features.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            out = model(inp, cam_feat).squeeze().cpu().tolist()
        
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
st.set_page_config(layout="wide", page_title="Object Dimension Estimator with Camera Features")
st.title("🎯 Object Dimension & Volume Estimation")
st.markdown("### Enhanced with Camera Metadata Features")

# Load models
dim_model, uses_camera_features = load_dimension_model()
d2_predictor, d2_cfg = (None, None)
d2_metadata = None
if d2_imported_successfully:
    d2_predictor, d2_cfg = load_detectron2_model()
    if d2_cfg:
        try:
            d2_metadata = MetadataCatalog.get(d2_cfg.DATASETS.TRAIN[0])
        except:
            d2_metadata = MetadataCatalog.get("coco_2017_val")

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    use_camera_metadata = st.checkbox(
        "Use Camera Metadata", 
        value=uses_camera_features,
        help="Include camera EXIF data (focal length, ISO, aperture, etc.) for improved estimation"
    )
    
    st.divider()
    st.subheader("📊 System Status")
    st.write(f"**Device:** {DEVICE}")
    st.write(f"**Detectron2:** {'✅ OK' if d2_predictor else '❌ Failed'}")
    st.write(f"**Dimension CNN:** {'✅ OK' if dim_model else '❌ Failed'}")
    st.write(f"**Camera Features:** {'✅ Enabled' if uses_camera_features else '⚠️ Not Available'}")
    
    if uses_camera_features:
        st.info("📸 Model using 10 camera features:\n- Focal Length\n- Aperture\n- ISO\n- Exposure Time\n- Image Dimensions\n- Aspect Ratio\n- Flash\n- White Balance\n- Exposure Mode")

# Main content
uploaded = st.file_uploader("📸 Upload an image", type=["jpg","jpeg","png"])

if uploaded:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"📷 {uploaded.name}")
        
        try:
            # Extract camera features first
            uploaded.seek(0)
            camera_feature_vector, raw_metadata = extract_camera_features(uploaded)
            
            # Load image
            uploaded.seek(0)
            img = Image.open(uploaded).convert("RGB")
            img_np = np.array(img)
            bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            
            # Display original image
            st.image(img_np, use_column_width=True, caption="Original Image")
            
        except Exception as e:
            st.error(f"Invalid image: {e}")
            bgr = None
            camera_feature_vector = None
    
    with col2:
        st.subheader("📊 Camera Metadata")
        
        if 'error' not in raw_metadata:
            with st.expander("📸 Camera Info", expanded=True):
                st.markdown(f"""
                **Device:**
                - Make: {raw_metadata['make']}
                - Model: {raw_metadata['model']}
                
                **Settings:**
                - Focal Length: {raw_metadata['focal_length_mm']}
                - Aperture: {raw_metadata['aperture_f']}
                - ISO: {raw_metadata['iso']}
                - Exposure: {raw_metadata['exposure_time_s']}
                - Flash: {raw_metadata['flash']}
                
                **Image:**
                - Resolution: {raw_metadata['width']}x{raw_metadata['height']}
                - DateTime: {raw_metadata['datetime']}
                """)
            
            if use_camera_metadata and uses_camera_features:
                with st.expander("🔢 Extracted Features"):
                    feature_names = [
                        'Focal Length', 'Aperture', 'ISO', 'Exposure Time',
                        'Width', 'Height', 'Aspect Ratio', 'Flash',
                        'White Balance', 'Exposure Mode'
                    ]
                    for name, val in zip(feature_names, camera_feature_vector.tolist()):
                        st.write(f"**{name}:** {val:.3f}")
        else:
            st.warning("⚠️ No camera metadata found")
            st.write(raw_metadata.get('error', 'Unknown error'))

    # Process image
    if bgr is not None and d2_predictor and dim_model:
        with st.spinner("🔍 Detecting objects and estimating dimensions..."):
            # Detectron2 prediction
            outs = d2_predictor(bgr)
            inst = outs["instances"].to("cpu")
            
            if len(inst)==0:
                st.warning("⚠️ No objects detected in the image.")
            else:
                # Visualize detections
                st.subheader("🎯 Detection Results")
                viz = Visualizer(bgr[:,:,::-1], metadata=d2_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
                out_vis = viz.draw_instance_predictions(inst)
                det_img = out_vis.get_image()[:,:,::-1]
                st.image(det_img, use_column_width=True, caption=f"Detected {len(inst)} object(s)")

                # Process largest object
                idx = get_largest_instance_index(inst)
                if idx>=0:
                    mask = inst[idx].pred_masks[0] if inst.has("pred_masks") else None
                    crop = crop_from_mask(img_np, mask) if mask is not None else None
                    
                    if crop is not None:
                        st.subheader("📏 Dimension Analysis")
                        
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            st.image(crop, caption="Cropped Object", use_column_width=True)
                        
                        with col2:
                            # Predict dimensions with or without camera features
                            if use_camera_metadata and uses_camera_features and camera_feature_vector is not None:
                                st.info("✅ Using camera metadata for enhanced estimation")
                                dims = predict_dimensions_cnn(crop, dim_model, camera_feature_vector, True)
                            else:
                                if not uses_camera_features:
                                    st.warning("⚠️ Model doesn't support camera features - using image only")
                                dims = predict_dimensions_cnn(crop, dim_model, None, False)
                            
                            # Display results in nice format
                            st.markdown("### 📐 Estimated Dimensions")
                            
                            metric_cols = st.columns(2)
                            with metric_cols[0]:
                                st.metric("Length", dims.get("Length (cm)", "N/A"))
                                st.metric("Width", dims.get("Width (cm)", "N/A"))
                            with metric_cols[1]:
                                st.metric("Height", dims.get("Height (cm)", "N/A"))
                                st.metric("Volume", dims.get("Volume (cm³)", "N/A"))
                            
                            # Export button
                            export_data = {
                                "filename": uploaded.name,
                                "timestamp": datetime.now().isoformat(),
                                "dimensions": dims,
                                "camera_metadata": raw_metadata if 'error' not in raw_metadata else None,
                                "used_camera_features": use_camera_metadata and uses_camera_features
                            }
                            
                            st.download_button(
                                "💾 Download Results (JSON)",
                                data=json.dumps(export_data, indent=2),
                                file_name=f"dimensions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                    else:
                        st.error("❌ Could not crop object from image.")
    elif not d2_imported_successfully:
        st.error("❌ Detectron2 not loaded.")
    else:
        st.error("❌ Models not loaded properly.")
else:
    # Landing page
    st.info("👆 Upload an image to get started!")
    
    st.markdown("""
    ### 🎯 Features:
    
    - **🤖 Object Detection**: Automatic detection using Detectron2 Mask R-CNN
    - **📏 Dimension Estimation**: ML-based estimation of Length, Width, Height, and Volume
    - **📸 Camera Metadata Integration**: Uses EXIF data to improve predictions:
        - Focal Length
        - Aperture (F-number)
        - ISO Speed
        - Exposure Time
        - Image Resolution
        - Flash Settings
        - White Balance
        - Exposure Mode
    - **🎨 Instance Segmentation**: Precise object masking and cropping
    - **💾 Export Results**: Download dimensions and metadata as JSON
    
    ### 📖 How It Works:
    
    1. Upload an image with camera metadata (EXIF)
    2. System extracts 10 camera features automatically
    3. Detectron2 detects and segments objects
    4. ResNet50 CNN estimates dimensions using both image and camera features
    5. View results with enhanced accuracy
    
    ### 💡 Tips:
    
    - Use images from real cameras (phones/DSLRs) for best metadata
    - Screenshots and edited images may lack EXIF data
    - The model can work without metadata but performs better with it
    - Clear, well-lit photos produce better results
    """)