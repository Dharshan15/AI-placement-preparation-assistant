import cv2
import mediapipe as mp
import speech_recognition as sr
import threading

# Flag to control listening
listening = True

# Initialize Mediapipe Face Detection and Drawing utilities
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def callback(recognizer, audio):
    global listening
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")

        # Save the recognized text to a text file
        with open("recognized_speech.txt", "a") as f:
            f.write(text + "\n")

        if "bye" in text.lower():
            print("Stopping listening...")
            listening = False
    except sr.UnknownValueError:
        print("Sorry, I could not understand your speech.")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

def speech_to_text_continuous():
    global listening
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        print("Listening for commands. Say 'bye' to stop.")
        while listening:
            try:
                audio = recognizer.listen(source, timeout=1)
                text = recognizer.recognize_google(audio)
                print(f"Recognized: {text}")

                # Save the recognized text to a text file
                with open("recognized_speech.txt", "a") as f:
                    f.write(text + "\n")

                if "bye" in text.lower():
                    print("Stopping listening...")
                    listening = False
            except sr.WaitTimeoutError:
                # Timeout without audio; continue listening
                continue
            except sr.UnknownValueError:
                # Handle unrecognized speech
                continue
            except sr.RequestError as e:
                print(f"Request error: {e}")
                break

def monitor_posture():
    cap = cv2.VideoCapture(0)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and detect faces
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(frame, detection)
                    # Extract the bounding box and key points
                    bbox = detection.location_data.relative_bounding_box
                    # Simple check if the face is centered (this can be customized)
                    if bbox.xmin > 0.3 and bbox.xmin < 0.7 and bbox.ymin > 0.3 and bbox.ymin < 0.7:
                        cv2.putText(frame, "Good Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "Bad Posture", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the frame with posture indication
            cv2.imshow('Posture Monitor', frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Create and start the speech recognition thread
    speech_thread = threading.Thread(target=speech_to_text_continuous)
    speech_thread.start()

    # Create and start the posture monitoring thread
    posture_thread = threading.Thread(target=monitor_posture)
    posture_thread.start()

    # Wait for both threads to finish
    try:
        speech_thread.join()
    except KeyboardInterrupt:
        listening = False

    posture_thread.join()
    print("Application stopped.")