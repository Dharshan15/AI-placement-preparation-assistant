import speech_recognition as sr
import re
from config import GEMINI_API_KEY
import google.generativeai as genai
import threading
import tkinter as tk
from tkinter import messagebox
import json
from json.decoder import JSONDecodeError
import cv2
import mediapipe as mp
import numpy as np
from fer import FER

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

class CombinedInterviewAnalyzer:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API key is missing.")
        self.recognizer = sr.Recognizer()
        self.listening = True
        self.current_answer = ""
        self.evaluation_result = None

        # Initialize mediapipe pose and face mesh estimators, and emotion detector
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        self.emotion_detector = FER(mtcnn=True)

        self.posture_score = 0.0
        self.emotion_score = 0.0

    def generate_questions(self, role, skill_level):
        import time  # Import time for timestamp
        prompt = f"Generate 10 interview questions for a {role} position at {skill_level} skill level. Number each question and provide only the questions without any additional text. Timestamp: {time.time()}"  # Added timestamp
        try:
            response = model.generate_content(prompt)
            questions = response.text.split('\n')
            questions = [q.strip() for q in questions if q.strip() and q[0].isdigit()]
            if not questions:
                print("No valid questions were generated. Trying again...")
                return self.generate_questions(role, skill_level)
            return questions[:10]
        except Exception as e:
            
            print(f"Error generating questions: {e}")
            return []

    def speech_to_text_continuous(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source)
            print("Listening for your answer...")
            while self.listening:
                try:
                    audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=10)
                    text = self.recognizer.recognize_google(audio)
                    print(f"Recognized: {text}")
                    self.current_answer += " " + text
                except sr.WaitTimeoutError:
                    continue
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    print(f"Request error: {e}")
                    break

    def get_voice_input(self):
        self.current_answer = ""
        self.listening = True
        
        # Start the speech recognition in a separate thread
        thread = threading.Thread(target=self.speech_to_text_continuous)
        thread.start()

    def conduct_interview(self):
        # Clear previous interview data
        open("interview_qa.txt", "w").close()
        open("interview_evaluation.json", "w").close()

        self.root = tk.Tk()
        self.root.title("Plaprep.ai")
        self.root.geometry("1200x800")

        self.setup_ui()

        self.root.mainloop()

    def setup_ui(self):
        # Left panel for interview
        left_panel = tk.Frame(self.root, width=600)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        role_label = tk.Label(left_panel, text="Enter the role for the interview:")
        role_label.pack(pady=10)
        self.role_entry = tk.Entry(left_panel, width=50)
        self.role_entry.pack(pady=5)

        skill_label = tk.Label(left_panel, text="Select the skill level:")
        skill_label.pack(pady=10)
        self.skill_var = tk.StringVar(value="beginner")  # Default value
        self.skill_dropdown = tk.OptionMenu(left_panel, self.skill_var, "beginner", "intermediate", "expert")
        self.skill_dropdown.pack(pady=5)

        self.start_button = tk.Button(left_panel, text="Start Interview", command=self.start_interview)
        self.start_button.pack(pady=20)

        self.question_label = tk.Label(left_panel, text="", wraplength=550, justify="center")
        self.question_label.pack(pady=20)

        self.next_button = tk.Button(left_panel, text="Next Question", command=self.next_question)
        self.next_button.pack(pady=10)
        self.next_button.pack_forget()

        # Right panel for video feed
        right_panel = tk.Frame(self.root, width=600)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(right_panel)
        self.video_label.pack(pady=20)

        self.posture_label = tk.Label(right_panel, text="Posture Score: 0.00")
        self.posture_label.pack(pady=10)

        self.emotion_label = tk.Label(right_panel, text="Emotion Score: 0.00")
        self.emotion_label.pack(pady=10)

    def start_interview(self):
        role = self.role_entry.get().strip()
        skill_level = self.skill_var.get().strip()

        if not role or not skill_level:
            messagebox.showerror("Error", "Please enter both role and skill level.")
            return

        self.questions = self.generate_questions(role, skill_level)
        if not self.questions:
            messagebox.showerror("Error", "Failed to generate questions. Please try again later.")
            return

        self.current_question_index = 0
        
        # Hide the input fields and start button
        self.role_entry.pack_forget()
        self.start_button.pack_forget()

        self.next_button.pack()
        self.show_question()

        # Start the video analysis
        self.start_video_analysis()

    def show_question(self):
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.question_label.config(text=f"Question{question}")
            self.get_voice_input()
        else:
            self.finish_interview()

    def next_question(self):
        self.listening = False
        answer = self.current_answer.strip()
        question = self.questions[self.current_question_index]

        if not answer:
            answer = "No answer provided"

        with open("interview_qa.txt", "a") as f:
            # f.write(f"Question {self.current_question_index + 1}: {question}\n")
            f.write(f"Question{question}\n")
            f.write(f"Answer: {answer}\n\n")

        self.current_question_index += 1
        self.show_question()

    def finish_interview(self):
        self.video_capture.release()
        self.root.withdraw()  # Hide the main window instead of quitting
        self.evaluate_answers_with_gemini()

    def evaluate_answers_with_gemini(self):
        with open("interview_qa.txt", "r") as f:
            interview_content = f.read()

        prompt = f"""
        You are an expert interviewer. Please evaluate the following interview questions and answers:

        {interview_content}

        For each question-answer pair, provide:
        1. A score out of 10
        2. A brief explanation for the score
        3. Suggestions for improvement

        Finally, provide an overall score out of 100 and general feedback on the interview performance.

        Format your response as a JSON object with the following structure:
        {{
            "evaluations": [
                {{
                    "question_number": 1,
                    "score": 8,
                    "explanation": "...",
                    "suggestions": "..."
                }},
                ...
            ],
            "overall_score": 85,
            "general_feedback": "..."
        }}
        """

        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Try to find JSON content within the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != -1:
                json_content = response_text[json_start:json_end]
                
                # Attempt to fix common JSON errors
                json_content = re.sub(r',\s*}', '}', json_content)  # Remove trailing commas
                json_content = re.sub(r',\s*]', ']', json_content)  # Remove trailing commas in arrays
                json_content = json_content.replace('```json', '').replace('```', '')  # Remove markdown code block syntax
                
                evaluation_result = json.loads(json_content)
            else:
                raise JSONDecodeError("No valid JSON found in the response", response_text, 0)

            with open("interview_evaluation.json", "w") as f:
                json.dump(evaluation_result, f, indent=2)

            self.evaluation_result = evaluation_result
            self.display_evaluation_results(evaluation_result)
        except JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print("Raw response:", response_text)
            messagebox.showerror("Evaluation Error", "Failed to parse the evaluation results. The raw response has been logged.")
        except Exception as e:
            print(f"Error evaluating answers: {e}")
            print("Raw response:", response.text)
            messagebox.showerror("Evaluation Error", "An error occurred while evaluating the answers. Please check the logs for more information.")

    def display_evaluation_results(self, evaluation_result):
        result_window = tk.Toplevel(self.root)
        result_window.title("Interview Evaluation Results")
        result_window.geometry("800x600")

        text_widget = tk.Text(result_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(text_widget)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=text_widget.yview)

        text_widget.insert(tk.END, f"Overall Score: {evaluation_result['overall_score']}/100\n\n")
        text_widget.insert(tk.END, f"General Feedback:\n{evaluation_result['general_feedback']}\n\n")

        for eval in evaluation_result['evaluations']:
            text_widget.insert(tk.END, f"Question {eval['question_number']}:\n")
            text_widget.insert(tk.END, f"Score: {eval['score']}/10\n")
            text_widget.insert(tk.END, f"Explanation: {eval['explanation']}\n")
            text_widget.insert(tk.END, f"Suggestions: {eval['suggestions']}\n\n")

        text_widget.config(state=tk.DISABLED)

        # Add a close button
        close_button = tk.Button(result_window, text="Close", command=self.close_application)
        close_button.pack(pady=10)

    def close_application(self):
        self.root.quit()
        self.root.destroy()

    def start_video_analysis(self):
        self.video_capture = cv2.VideoCapture(0)
        self.analyze_video()

    def analyze_video(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Analyze the frame
            frame = cv2.resize(frame, (640, 480))
            self.posture_score, self.emotion_score, pose_results, face_results = self.analyze_frame(frame)

            # Update UI with scores
            self.posture_label.config(text=f"Posture Score: {self.posture_score:.2f}")
            self.emotion_label.config(text=f"Emotion Score: {self.emotion_score:.2f}")

            # Display the frame with feedback
            self.display_feedback(frame, pose_results, face_results)

            # Schedule the next frame analysis
            self.root.after(50, self.analyze_video)

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

    def display_feedback(self, frame, pose_results, face_results):
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

        # Convert the frame to RGB for displaying in tkinter
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = tk.PhotoImage(data=cv2.imencode('.png', rgb_frame)[1].tobytes())
        
        # Update the video feed in the UI
        self.video_label.config(image=img)
        self.video_label.image = img  # Keep a reference to prevent garbage collection

if __name__ == "__main__":
    analyzer = CombinedInterviewAnalyzer()
    analyzer.conduct_interview()