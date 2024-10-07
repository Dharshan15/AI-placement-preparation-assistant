import speech_recognition as sr
import re
from config import GEMINI_API_KEY
import google.generativeai as genai
import threading
import tkinter as tk
from tkinter import messagebox
import json
from json.decoder import JSONDecodeError

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

class QAEvaluator:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API key is missing.")
        self.recognizer = sr.Recognizer()
        self.listening = True
        self.current_answer = ""
        self.evaluation_result = None

    def generate_questions(self, role, skill_level):
        prompt = f"Generate 10 interview questions for a {role} position at {skill_level} skill level. Number each question and provide only the questions without any additional text."
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
        self.root.title("Mock Interview")
        self.root.geometry("600x400")

        self.setup_ui()

        self.root.mainloop()

    def setup_ui(self):
        role_label = tk.Label(self.root, text="Enter the role for the interview:")
        role_label.pack(pady=10)
        self.role_entry = tk.Entry(self.root, width=50)
        self.role_entry.pack(pady=5)

        skill_label = tk.Label(self.root, text="Enter the skill level (e.g., beginner, intermediate, expert):")
        skill_label.pack(pady=10)
        self.skill_entry = tk.Entry(self.root, width=50)
        self.skill_entry.pack(pady=5)

        self.start_button = tk.Button(self.root, text="Start Interview", command=self.start_interview)
        self.start_button.pack(pady=20)

        self.question_label = tk.Label(self.root, text="", wraplength=550, justify="center")
        self.question_label.pack(pady=20)

        self.next_button = tk.Button(self.root, text="Next Question", command=self.next_question)
        self.next_button.pack(pady=10)
        self.next_button.pack_forget()  

    def start_interview(self):
        role = self.role_entry.get().strip()
        skill_level = self.skill_entry.get().strip()

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
        self.skill_entry.pack_forget()
        self.start_button.pack_forget()

        self.next_button.pack()
        self.show_question()

    def show_question(self):
        if self.current_question_index < len(self.questions):
            question = self.questions[self.current_question_index]
            self.question_label.config(text=f"Question {self.current_question_index + 1}: {question}")
            self.get_voice_input()
        else:
            self.finish_interview()

        # Update the score if available
        if self.evaluation_result and self.current_question_index > 0:
            current_score = self.evaluation_result['evaluations'][self.current_question_index - 1]['score']
            self.update_score_display(current_score)

    def update_score_display(self, score):
        if hasattr(self, 'score_label'):
            self.score_label.config(text=f"Previous Question Score: {score}/10")
        else:
            self.score_label = tk.Label(self.root, text=f"Previous Question Score: {score}/10")
            self.score_label.pack(pady=10)

    def next_question(self):
        self.listening = False
        answer = self.current_answer.strip()
        question = self.questions[self.current_question_index]

        if not answer:
            answer = "No answer provided"

        with open("interview_qa.txt", "a") as f:
            f.write(f"Question {self.current_question_index + 1}: {question}\n")
            f.write(f"Answer: {answer}\n\n")

        self.current_question_index += 1
        self.show_question()

    def finish_interview(self):
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

if __name__ == "__main__":
    evaluator = QAEvaluator()
    evaluator.conduct_interview()