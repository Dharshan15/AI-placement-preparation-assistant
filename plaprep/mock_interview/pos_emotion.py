import cv2
import mediapipe as mp
import numpy as np
from fer import FER

class PostureEmotionAnalyzer:
    def __init__(self):
        # Initialize mediapipe pose and face mesh estimators, and emotion detector
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh()
        self.emotion_detector = FER(mtcnn=True)

    def analyze_frame(self, frame):
        # Convert the frame to RGB for mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Analyze posture using mediapipe
        pose_results = self.pose.process(rgb_frame)
        
        # Analyze face direction using face mesh
        face_results = self.face_mesh.process(rgb_frame)
        
        # Analyze emotion using FER
        emotions = self.emotion_detector.detect_emotions(frame)
        
        # Evaluate posture and emotion to get scores
        posture_score = self.evaluate_posture(pose_results)
        emotion_score = self.evaluate_emotion(emotions)
        
        return posture_score, emotion_score, pose_results, face_results

    def evaluate_posture(self, pose_results):
        if not pose_results.pose_landmarks:
            return 0.0
        
        # Get landmarks only above the chest: shoulders, head, neck, etc.
        landmarks = pose_results.pose_landmarks.landmark
        shoulder_left = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
        shoulder_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
        wrist_left = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST]
        wrist_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        neck = landmarks[self.mp_pose.PoseLandmark.NOSE]  # Using nose as an approximation for neck

        # Initialize posture score
        posture_score = 1.0

        # Calculate forward/backward tilt
        vertical_vector = np.array([0, -1])  # Pointing upwards
        neck_to_head_vector = np.array([nose.x - neck.x, nose.y - neck.y])
        neck_to_head_norm = neck_to_head_vector / np.linalg.norm(neck_to_head_vector)
        
        forward_tilt = np.arccos(np.clip(np.dot(vertical_vector, neck_to_head_norm), -1.0, 1.0))
        forward_tilt_degrees = np.degrees(forward_tilt)

        # Penalize score for forward/backward tilt
        if forward_tilt_degrees > 20:  # Increased threshold
            tilt_penalty = min((forward_tilt_degrees - 20) / 40, 0.2)  # Reduced max penalty
            posture_score -= tilt_penalty

        # Calculate left/right tilt
        shoulder_vector = np.array([shoulder_right.x - shoulder_left.x, shoulder_right.y - shoulder_left.y])
        shoulder_norm = shoulder_vector / np.linalg.norm(shoulder_vector)
        horizontal_vector = np.array([1, 0])

        side_tilt = np.arccos(np.clip(np.dot(horizontal_vector, shoulder_norm), -1.0, 1.0))
        side_tilt_degrees = np.degrees(side_tilt)

        # Penalize score for left/right tilt
        if abs(side_tilt_degrees - 90) > 15:  # Increased threshold
            tilt_penalty = min((abs(side_tilt_degrees - 90) - 15) / 30, 0.2)  # Reduced max penalty
            posture_score -= tilt_penalty

        # Penalize if hands are raised abnormally (higher than shoulders)
        if wrist_left.y < shoulder_left.y or wrist_right.y < shoulder_right.y:
            posture_score -= 0.1  # Further reduced penalty for raised hands

        # Calculate shoulder symmetry
        shoulder_diff = abs(shoulder_left.y - shoulder_right.y)
        symmetry_penalty = min(shoulder_diff * 1.5, 0.2)  # Reduced max penalty and multiplier
        posture_score -= symmetry_penalty

        # Ensure the score is between 0 and 1
        return max(0, min(posture_score, 1))

    def evaluate_emotion(self, emotions):
        if not emotions:
            return 0.95  # Return max score if no emotions detected
        
        # Consider the first detected face's emotions
        emotion = emotions[0]["emotions"]
        anxiety_score = emotion.get("fear", 0)  # Assuming "fear" represents anxiety
        
        # Calculate emotion score (1 - anxiety_score)
        # This will reduce the score as anxiety increases
        emotion_score = 0.95 - anxiety_score
        
        return max(0, min(emotion_score, 1))  # Ensure score is between 0 and 1
    
    def display_feedback(self, frame, posture_score, emotion_score, pose_results, face_results):
        # Display posture feedback
        posture_text = f"Posture Score: {posture_score:.2f}"
        cv2.putText(frame, posture_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display emotion feedback
        emotion_text = f"Emotion Score: {emotion_score:.2f}"
        cv2.putText(frame, emotion_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Draw skeletal points (landmarks above chest)
        if pose_results.pose_landmarks:
            upper_body_landmarks = [
                self.mp_pose.PoseLandmark.NOSE, 
                self.mp_pose.PoseLandmark.LEFT_EYE, 
                self.mp_pose.PoseLandmark.RIGHT_EYE, 
                self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                self.mp_pose.PoseLandmark.RIGHT_SHOULDER
            ]
            for landmark in upper_body_landmarks:
                point = pose_results.pose_landmarks.landmark[landmark]
                frame_height, frame_width, _ = frame.shape
                cx, cy = int(point.x * frame_width), int(point.y * frame_height)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)

        # Draw facial landmarks to show face direction
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face.FACEMESH_CONTOURS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1)
                )
        
    def run(self):
        # Access the camera and perform analysis
        cap = cv2.VideoCapture(0)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Analyze the frame for posture, face direction, and emotion
            posture_score, emotion_score, pose_results, face_results = self.analyze_frame(frame)
            
            # Display feedback on the frame
            self.display_feedback(frame, posture_score, emotion_score, pose_results, face_results)

            # Show the frame with feedback
            cv2.imshow('Posture and Emotion Analyzer', frame)
            
            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# Usage example
if __name__ == "__main__":
    analyzer = PostureEmotionAnalyzer()
    analyzer.run()