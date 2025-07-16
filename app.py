import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

st.title("🧵 Real-Time Fabric Defect Detection")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
class_names = ['No Defect', 'Hole', 'Stain', 'Misweave']  # update your classes

# Function to validate frame
def is_valid_frame(frame):
    return frame is not None and frame.size > 0 and len(frame.shape) == 3

# Open webcam
run = st.checkbox('Start Camera')
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        st.error("❌ Could not access webcam. Please check your camera connection.")
        st.stop()
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while run:
        ret, frame = cap.read()
        if not ret or not is_valid_frame(frame):
            st.warning("⚠️ Could not read frame from webcam")
            break

        # Flip for natural webcam view
        frame = cv2.flip(frame, 1)
        
        try:
            # YOLO prediction - pass the raw frame directly
            # YOLO will handle its own preprocessing
            results = model.predict(frame, verbose=False, conf=0.5)
            
            # Process results
            if len(results) > 0 and len(results[0].boxes) > 0:
                # Get the highest confidence detection
                boxes = results[0].boxes
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                
                # Find highest confidence detection
                max_conf_idx = np.argmax(confidences)
                detected_class = classes[max_conf_idx]
                confidence = confidences[max_conf_idx]
                
                # Get class name
                if detected_class < len(class_names):
                    label = class_names[detected_class]
                else:
                    label = f"Class {detected_class}"
                
                # Display prediction with confidence
                text = f'Defect: {label} ({confidence:.2f})'
                color = (0, 255, 0) if label == 'No Defect' else (0, 0, 255)
                
                # Draw bounding box if available
                if len(boxes.xyxy) > 0:
                    box = boxes.xyxy[max_conf_idx].cpu().numpy().astype(int)
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    
                    # Put text above the box
                    cv2.putText(frame, text, (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    # If no bounding box, put text at top
                    cv2.putText(frame, text, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            else:
                # No detections
                cv2.putText(frame, 'No Defect Detected', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            cv2.putText(frame, 'Prediction Error', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Show result
        FRAME_WINDOW.image(frame, channels="BGR")

    cap.release()
else:
    st.info("👆 Check the box above to start the camera")