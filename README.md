# Counterfeit Currency Identification Agent

FastAPI + OpenCV prototype for Rs 500 note authenticity screening across microprint texture, security thread continuity, serial number OCR, and simulated UV feature visibility.

## Run locally

```powershell
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000/

## API

```text
POST /api/v1/scan
Content-Type: multipart/form-data
field: file
```

## Docker

```powershell
docker build -t counterfeit-currency-agent .
docker run --rm -p 8000:8000 counterfeit-currency-agent
```

Open http://127.0.0.1:8000/

## EasyOCR model behavior

By default, the app uses `download_enabled=False` for offline-first operation. If a cloud deployment needs to download EasyOCR's English model automatically, set:

```text
EASYOCR_DOWNLOAD_ENABLED=true
```

For a strict offline deployment, pre-cache the EasyOCR English model in the runtime image or leave OCR fallback warnings enabled.
