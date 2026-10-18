import cv2
from ultralytics import YOLO
import threading
import os

model = YOLO("yolo11n.pt")
camera = cv2.VideoCapture(0)

last_spoken = ""
speaking = False


def speak(text):
    global speaking

    speaking = True

    os.system(
        f'powershell -Command "Add-Type -AssemblyName System.Speech; '
        f'(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak(\'{text}\')"'
    )

    speaking = False


while True:

    success, frame = camera.read()

    if not success:
        print("Camera could not be opened")
        break

    results = model(frame)

    if len(results[0].boxes) > 0:

     detected_objects = []

    for box in results[0].boxes:
        detected = results[0].names[int(box.cls[0])]

        if detected not in detected_objects:
            detected_objects.append(detected)

    for detected in detected_objects:

        if detected != last_spoken and not speaking:

            print("SPEAKING:", detected)

            last_spoken = detected

            threading.Thread(
                target=speak,
                args=(detected,),
                daemon=True
            ).start()

    annotated_frame = results[0].plot()

    cv2.imshow("ARGUS - Object Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()