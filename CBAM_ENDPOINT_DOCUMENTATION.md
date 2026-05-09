# YOLOv12 CBAM Endpoint Documentation

## 📋 Ringkasan Perubahan

Telah ditambahkan support untuk **YOLOv12 dengan CBAM (Convolutional Block Attention Module)** ke dalam API v2.0. Model ini dimuat dari path:

- **CBAM Model Path:** `D:\UMNFIX\yolov12_aca\runs\detect\train4\weights\best.pt`

## 🔧 Konfigurasi Model

### Konstanta Baru

```python
CBAM_MODEL_PATH = os.getenv(
    "YOLO_CBAM_MODEL_PATH",
    r"D:\UMNFIX\yolov12_aca\runs\detect\train4\weights\best.pt",
)
```

### Model Manager

Ditambahkan instance baru:

```python
cbam_model_manager = ModelManager(CBAM_MODEL_PATH)
```

## 🚀 Endpoint CBAM yang Ditambahkan

### 1. **Single Image Detection (CBAM)**

**Endpoint:** `POST /v1/detect/cbam`

**Deskripsi:** Deteksi penyakit mata menggunakan YOLOv12 dengan CBAM Attention Module

**Parameters:**

- `file` (UploadFile): Image file untuk dianalisis
- `conf` (float, 0.1-1.0): Confidence threshold (default: 0.25)
- `iou` (float, 0.1-1.0): IOU threshold untuk NMS (default: 0.45)
- `return_image` (bool): Return annotated image dalam base64 (default: false)

**Response:**

```json
{
  "success": true,
  "message": "CBAM Deteksi selesai: 3 penyakit ditemukan",
  "timestamp": "2026-04-26T10:00:00",
  "image_metadata": {
    "filename": "eye_image.jpg",
    "size": 12345,
    "width": 1920,
    "height": 1080,
    "channels": 3,
    "model": "YOLOv12 CBAM"
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "Disease1",
      "confidence": 0.95,
      "bbox": { "x1": 100, "y1": 100, "x2": 200, "y2": 200 },
      "bbox_normalized": { "x1": 0.052, "y1": 0.093, "x2": 0.104, "y2": 0.185 },
      "area_pixels": 10000
    }
  ],
  "statistics": {
    "total_detections": 3,
    "average_confidence": 0.92,
    "min_confidence": 0.88,
    "max_confidence": 0.96,
    "image_area_pixels": 2073600,
    "detection_density": 0.00144
  },
  "image_base64": null
}
```

---

### 2. **Batch Detection (CBAM)**

**Endpoint:** `POST /v1/detect/cbam/batch`

**Deskripsi:** Batch processing untuk multiple images menggunakan CBAM model

**Parameters:**

- `files` (List[UploadFile]): List of image files (max 50 files)
- `conf` (float): Confidence threshold
- `iou` (float): IOU threshold

**Response:**

```json
{
  "success": true,
  "message": "CBAM Batch processing selesai: 10/10 berhasil",
  "timestamp": "2026-04-26T10:00:00",
  "total_images": 10,
  "processed_images": 10,
  "results": [
    {
      "filename": "image1.jpg",
      "success": true,
      "detections": 5,
      "image_shape": { "width": 1920, "height": 1080 },
      "model": "YOLOv12 CBAM"
    }
  ]
}
```

---

## 📡 Updated Endpoints

### Health Check (Updated)

**Endpoint:** `GET /v1/health`

Sekarang mencakup status kedua model (Standard dan CBAM):

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda",
  "model_info": {
    "standard": {
      "loaded": true,
      "device": "cuda",
      "num_classes": 5
    },
    "cbam": {
      "loaded": true,
      "device": "cuda",
      "num_classes": 5
    }
  },
  "timestamp": "2026-04-26T10:00:00"
}
```

### Model Info (Updated)

**Endpoint:** `GET /v1/model/info`

Menampilkan informasi detail tentang kedua model:

```json
{
  "success": true,
  "models": {
    "standard": {
      "name": "YOLOv12 with Attention Mechanism",
      "task": "Object Detection",
      "path": "...",
      "device": "cuda",
      "num_classes": 5,
      "classes": {...}
    },
    "cbam": {
      "name": "YOLOv12 ACA with CBAM",
      "task": "Object Detection",
      "attention": "CBAM (Convolutional Block Attention Module)",
      "path": "D:\\UMNFIX\\yolov12_aca\\runs\\detect\\train4\\weights\\best.pt",
      "device": "cuda",
      "num_classes": 5,
      "classes": {...}
    }
  },
  "endpoints": {
    "standard": {
      "detect": "/v1/detect",
      "batch": "/v1/detect/batch"
    },
    "cbam": {
      "detect": "/v1/detect/cbam",
      "batch": "/v1/detect/cbam/batch"
    }
  },
  "timestamp": "2026-04-26T10:00:00"
}
```

### Root Endpoint (Updated)

**Endpoint:** `GET /`

Sekarang menampilkan semua endpoint termasuk CBAM endpoints.

---

## 🧠 CBAM Attention Module

Modul CBAM terdiri dari:

### 1. Channel Attention

```python
class ChannelAttention(torch.nn.Module):
    """Channel Attention module"""
    - Menggunakan AdaptiveAvgPool2d dan AdaptiveMaxPool2d
    - MLP dengan Conv2d untuk channel-wise attention
    - Output: Channel attention map dengan shape (B, C, 1, 1)
```

### 2. Spatial Attention

```python
class SpatialAttention(torch.nn.Module):
    """Spatial Attention module"""
    - Menggabungkan average pooling dan max pooling across channel
    - Conv2d dengan kernel size 7x7 untuk spatial attention
    - Output: Spatial attention map dengan shape (B, 1, H, W)
```

### 3. CBAM Module

```python
class CBAM(torch.nn.Module):
    """CBAM module menggabungkan Channel dan Spatial Attention"""
    def forward(self, x):
        x = x * self.channel_attention(x)  # Channel attention
        x = x * self.spatial_attention(x)   # Spatial attention
        return x
```

### 4. CBAM Wrapper

```python
class CBAMWrapper(torch.nn.Module):
    """Wrapper untuk integrasi CBAM ke dalam YOLOv12 architecture"""
    - Wrap base module dengan CBAM layer
    - Lazy initialization di forward pass
    - Kompatibel dengan YOLO's persistent hooks
```

---

## 📝 Contoh Penggunaan

### 1. Single Image Detection dengan cURL

```bash
curl -X POST "http://localhost:8000/v1/detect/cbam" \
  -H "accept: application/json" \
  -F "file=@eye_image.jpg" \
  -F "conf=0.25" \
  -F "return_image=true"
```

### 2. Batch Detection dengan cURL

```bash
curl -X POST "http://localhost:8000/v1/detect/cbam/batch" \
  -H "accept: application/json" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "conf=0.25"
```

### 3. Python Client

```python
import requests

# Single image
files = {'file': open('eye_image.jpg', 'rb')}
params = {'conf': 0.25, 'return_image': True}
response = requests.post(
    'http://localhost:8000/v1/detect/cbam',
    files=files,
    params=params
)

# Batch
files = [
    ('files', open('image1.jpg', 'rb')),
    ('files', open('image2.jpg', 'rb')),
]
response = requests.post(
    'http://localhost:8000/v1/detect/cbam/batch',
    files=files,
    params={'conf': 0.25}
)
```

---

## 🔄 Model Loading Flow

1. **Inisialisasi API Server:**

   ```
   Model Manager dibuat untuk kedua paths:
   - Standard: BASE_DIR/runs/detect/train4/weights/best.pt
   - CBAM: D:/UMNFIX/yolov12_aca/runs/detect/train4/weights/best.pt
   ```

2. **Load Model dengan Custom Unpickler:**
   - SafeUnpickler menangani CBAM module serialization
   - Patched torch.load untuk support persistent_load
   - Automatic device detection (GPU/CPU)

3. **Inference:**
   - Input image di-preprocess
   - Forward pass melalui YOLOv12 + CBAM layers
   - Post-process detections dengan confidence dan IOU thresholds
   - Return structured results dengan metadata

---

## ⚙️ Environment Variables

```bash
# Set custom CBAM model path (optional)
set YOLO_CBAM_MODEL_PATH=D:\custom\path\to\model.pt

# Set standard model path (optional)
set YOLO_MODEL_PATH=D:\custom\path\to\model.pt
```

---

## 📊 Perbandingan Model

| Aspek         | Standard YOLOv12                            | YOLOv12 CBAM                                             |
| ------------- | ------------------------------------------- | -------------------------------------------------------- |
| **Path**      | BASE_DIR/runs/detect/train4/weights/best.pt | D:\UMNFIX\yolov12_aca\runs\detect\train4\weights\best.pt |
| **Attention** | AAttn (Legacy Attention)                    | CBAM                                                     |
| **Endpoint**  | /v1/detect, /v1/detect/batch                | /v1/detect/cbam, /v1/detect/cbam/batch                   |
| **Classes**   | 5 eye disease classes                       | 5 eye disease classes                                    |
| **Device**    | Auto-detect GPU/CPU                         | Auto-detect GPU/CPU                                      |

---

## 🐛 Troubleshooting

### Model tidak load

```python
# Check model path exists
import os
path = r"D:\UMNFIX\yolov12_aca\runs\detect\train4\weights\best.pt"
print(f"Model exists: {os.path.exists(path)}")

# Check device
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### CBAM Module pickle error

- SafeUnpickler akan otomatis handle custom CBAM modules
- Jika masih error, cek bahwa semua CBAM classes terdaftar di builtins

### Device mismatch

- API akan otomatis downgrade ke CPU jika GPU error
- Monitor logs di startup untuk device selection

---

## 📚 Referensi

- **CBAM Paper:** "CBAM: Convolutional Block Attention Module"
- **YOLOv12:** Ultralytics YOLO v12
- **Model Training:** Notebook di `YOLO12_FIX_BET_ANJ (2).ipynb`
- **CBAM Training Code:** Cells 1130-1365 notebook

---

**Last Updated:** 2026-04-26
**API Version:** 2.0.0
**Status:** ✅ Production Ready
