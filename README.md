# 💵 Counterfeit Currency Identification Agent

An AI-powered Computer Vision prototype for identifying counterfeit Indian ₹500 currency notes using image processing techniques. The system analyzes multiple security features of a currency note and generates an authenticity score with visual inspection results.

> **Note:** This project is developed as a hackathon/educational prototype to demonstrate Computer Vision, OCR, and Image Processing concepts. It is **not** an official currency authentication system.

---

# 🚀 Features

✅ Upload an image of an Indian ₹500 currency note

✅ Real-time camera capture support

✅ Multi-stage image processing pipeline

✅ Microprint texture analysis

✅ Security thread verification

✅ Serial number OCR validation

✅ UV feature simulation using image enhancement

✅ Authenticity score generation

✅ Visual bounding box annotations

✅ FastAPI REST API

✅ Fully offline operation (except optional EasyOCR model download)

---

# 🛠 Tech Stack

### Backend
- Python 3.11
- FastAPI
- OpenCV
- EasyOCR
- NumPy
- Pillow

### Frontend
- HTML5
- Tailwind CSS
- Vanilla JavaScript

### AI / Computer Vision
- Image Processing
- OCR
- Edge Detection
- HSV Color Segmentation
- CLAHE Contrast Enhancement
- Contour Analysis
- Regular Expression Validation

---

# 🏗 Project Architecture

```
                User
                  │
                  ▼
        Upload Image / Camera
                  │
                  ▼
             FastAPI Server
                  │
                  ▼
          CurrencyDetector
                  │
 ┌─────────────────────────────────────┐
 │                                     │
 │ 1. Microprint Analysis              │
 │ 2. Security Thread Detection        │
 │ 3. Serial Number OCR                │
 │ 4. UV Feature Simulation            │
 │                                     │
 └─────────────────────────────────────┘
                  │
                  ▼
        Authenticity Score Engine
                  │
                  ▼
            JSON Response
                  │
                  ▼
        Frontend Visualization
```

---

# 📂 Project Structure

```
counterfeit-currency-identification-agent/

│── detector.py          # Computer Vision pipeline
│── main.py              # FastAPI application
│── index.html           # Frontend dashboard
│── requirements.txt     # Python dependencies
│── Dockerfile
│── README.md
│── .gitignore
│── .dockerignore
```

---

# 🔍 Security Verification Pipeline

## 1️⃣ Microprint Analysis

Detects fine-grained printed texture using edge density and Laplacian variance.

Techniques:

- Edge Detection
- Texture Analysis
- Laplacian Variance

---

## 2️⃣ Security Thread Verification

Detects the RBI security thread by:

- HSV Color Segmentation
- Contour Detection
- Continuity Analysis

---

## 3️⃣ Serial Number Validation

Extracts the serial number using EasyOCR and validates its format using Regular Expressions.

---

## 4️⃣ UV Feature Simulation

Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to simulate enhanced visibility of hidden security features.

---

# 📊 Output

The API returns:

- Authenticity Score
- Individual Security Check Status
- Bounding Boxes
- OCR Result
- Detection Summary

Example:

```json
{
  "authenticity_score": 86,
  "vectors": [
    {
      "name": "microprint",
      "status": "passed"
    },
    {
      "name": "security_thread",
      "status": "passed"
    },
    {
      "name": "serial_number",
      "status": "failed"
    },
    {
      "name": "uv_features",
      "status": "passed"
    }
  ]
}
```

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/satya-anand-ml/counterfeit-currency-identification-agent.git
```

Go into the project

```bash
cd counterfeit-currency-identification-agent
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the server

```bash
uvicorn main:app --reload
```

Open

```
http://127.0.0.1:8000/
```

---

# 📡 API Endpoint

### Scan Currency Note

```
POST /api/v1/scan
```

Request

```
multipart/form-data

field:
file
```

Response

```
JSON
```

---

# 🐳 Docker

Build image

```bash
docker build -t counterfeit-currency-agent .
```

Run container

```bash
docker run -p 8000:8000 counterfeit-currency-agent
```

---

# 💡 Future Improvements

- Deep Learning based counterfeit classification
- Support for multiple Indian currency denominations
- Mobile application
- YOLO-based note localization
- Real UV image verification
- Edge AI deployment
- TensorRT optimization
- Explainable AI visualization

---

# 📚 Learning Outcomes

Through this project, I gained practical experience with:

- FastAPI REST APIs
- Computer Vision
- OpenCV
- OCR using EasyOCR
- Image Processing
- HSV Color Space
- CLAHE
- Contour Detection
- Python Backend Development
- Frontend Integration

---

# 👨‍💻 Author

**Satya Anand**

B.Tech Computer Science & Engineering

Haldia Institute of Technology

GitHub:

https://github.com/satya-anand-ml

---

# ⭐ If you found this project useful, consider giving it a Star!
