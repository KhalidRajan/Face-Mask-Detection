import cv2
import keras
import numpy as np


face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture=cv2.VideoCapture(0)

# Load our Model
model=keras.models.load_model('face_detect')


# Indefinite loop until user exits by pressing the "Q" Key
while True:
    x, frame=video_capture.read()

    faces=face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3)

    for (x,y,w,h) in faces:
        # Capture face in frame
        face_image=frame[y:y+h, x:x+w]
        # Resize image to have dimensions (150, 150)
        resized_image=cv2.resize(face_image, (150, 150))
        # Normalize Image
        normalized_image=np.array(resized_image)/255.
        # Reshape NumPy array
        reshaped_image=np.reshape(normalized_image, (1, 150, 150, 3))

        #Pass into our model to predict and output label
        result=model.predict(reshaped_image)
        label=np.argmax(result, axis=1)[0]

        if label==0:
            text="Mask"
            color=(0, 255, 0) # Green

            # Draw the rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            # Add text above Rectangle
            cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

        else:
            text="No Mask"
            color=(0, 0, 255) # Red
            # Draw the rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            # Add text above Rectangle
            cv2.putText(frame, text, (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)


    cv2.imshow("Live", frame)
    key=cv2.waitKey(1) & 0xFF

    # If q key is pressed exit
    if key == ord('q'):
        break
# Destroy all windows and end video capture
video_capture.release()
cv2.destroyAllWindows()