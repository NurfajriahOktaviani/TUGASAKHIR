# Deploy ke Railway

Gunakan folder ini (`D:\UMNFIX\yolov12`) sebagai repository GitHub atau sebagai Railway Root Directory.

## File penting

- `api_v2.py`: FastAPI app dan entrypoint Railway.
- `requirements-api-v2.txt`: dependency runtime.
- `railway.json`: build/start/healthcheck untuk Railway.
- `Procfile`: fallback start command.
- `runs/detect/train4/weights/best.pt`: model CBAM.
- `runs_eca/detect/train4/weights/best.pt`: model ECA.

## Railway

Jika repository GitHub berisi folder ini sebagai root, Railway akan otomatis memakai:

```bash
python api_v2.py
```

Jika repository GitHub berisi folder parent `D:\UMNFIX`, set Railway Root Directory ke:

```text
yolov12
```

## Environment variables opsional

```text
ENABLE_CBAM_MODEL=true
ENABLE_ECA_MODEL=true
YOLO_MODEL_PATH=runs/detect/train4/weights/best.pt
YOLO_ECA_MODEL_PATH=runs_eca/detect/train4/weights/best.pt
```

Railway akan mengisi `PORT` otomatis. Jangan hard-code port di dashboard.

## Test setelah deploy

```bash
curl https://<domain-railway>/v1/health
```

