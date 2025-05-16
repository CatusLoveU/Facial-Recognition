# Face Mask Detection

This project uses TensorFlow and OpenCV to detect whether a person is wearing a face mask in real-time using a webcam.

---

## 📦 Requirements

- Python 3.8+
- TensorFlow
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

---

## ⚙️ Setup Instructions

### Step 1: Create and Activate Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 2: Install Required Libraries

pip install tensorflow opencv-python numpy scikit-learn matplotlib

### Step 3: Train the model

python train_model.py

⚠️ Note:

If mask_detector.h5 already exists, you can skip this step.

If the model doesn't work properly, delete the mask_detector.h5 file and re-run the training script.

### Step 4: Run the Application

python main.py
