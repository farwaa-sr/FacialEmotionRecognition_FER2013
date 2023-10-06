import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load JSON and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# Load weights into the model
emotion_model.load_weights("model/emotion_model.h5")
print("Model loaded from disk")

# Start the webcam feed
cap = cv2.VideoCapture(0)

emotion_report = []  # Variable to store emotion predictions

while True:
    # Read frame from the webcam feed
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    face_detector_front = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame_front = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    num_faces = face_detector_front.detectMultiScale(gray_frame_front, scaleFactor=1.3, minNeighbors=5)

    # Process each face detected
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame_front = gray_frame_front[y:y + h, x:x + w]
        cropped_img_front = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame_front, (48, 48)), -1), 0)

        # Predict the emotions
        emotion_prediction_front = emotion_model.predict(cropped_img_front)
        maxindex_front = int(np.argmax(emotion_prediction_front))
        cv2.putText(frame, emotion_dict[maxindex_front], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Append emotion to the report
        emotion_report.append(emotion_dict[maxindex_front])

    

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Generate Emotion Report
emotion_count = {}
total_emotions = len(emotion_report)

for emotion in emotion_report:
    if emotion in emotion_count:
        emotion_count[emotion] += 1
    else:
        emotion_count[emotion] = 1

emotion_percentage = {}
for emotion, count in emotion_count.items():
    percentage = (count / total_emotions) * 100
    emotion_percentage[emotion] = percentage

print("Emotion Report:")
for emotion, percentage in emotion_percentage.items():
    print(f"{emotion}: {percentage:.2f}%")