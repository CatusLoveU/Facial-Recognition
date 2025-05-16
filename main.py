import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

# Download mask detection model and face classifier
model = tf.keras.models.load_model("mask_detector.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
CATEGORIES = ["Có khẩu trang", "Không có khẩu trang", "Đeo khẩu trang không đúng cách"]

# Path to Vietnamese supported font
font_path = "F:\Learning\MachineLearning\BacTu\BacTu\Roboto-Black.ttf"
font = ImageFont.truetype(font_path, 20)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể truy cập webcam!")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (224, 224))
        face_resized = face_resized.astype("float32") / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        (mask, no_mask, mask_incorrect) = model.predict(face_resized)[0]
        
        # Check if mask is not worn or not worn properly
        if no_mask > mask and no_mask > mask_incorrect:
            label = 1  
            label_text = CATEGORIES[label]
            color = (0, 0, 255)  
            label_with_prob = f"{label_text}"
        elif mask < 0.7:
            label = 2  
            label_text = CATEGORIES[label]
            color = (0, 165, 255)  
            label_with_prob = f"{label_text}"
        else:
            label = 0
            label_text = CATEGORIES[label]
            color = (0, 255, 0)
            label_with_prob = f"{label_text}"
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.text((x, y - 30), label_with_prob, font=font, fill=color[::-1])  

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Phat Hien Khau Trang', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: 
        break

cap.release()
cv2.destroyAllWindows()
