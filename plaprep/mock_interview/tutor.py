from config import GEMINI_API_KEY
import google.generativeai as genai
import streamlit as st

# Configure the generative AI model
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

class TutorApp:
    def __init__(self):
        st.title("AI Tutor")

        # Define topics (for future extension)
        self.topics = {
            "Data Structures and Algorithms (DSA)": "Learn about arrays, linked lists, trees, etc.",
            "Object-Oriented Programming (OOP)": "Understand classes, objects, inheritance, etc.",
            "Operating Systems (OS)": "Explore processes, threads, memory management, etc.",
            "Databases": "Study SQL, NoSQL, normalization, etc.",
            "Computer Networks": "Learn about protocols, OSI model, etc."
        }

        # Initialize chat history in session state if not present
        if 'gemini_chat' not in st.session_state:
            st.session_state.gemini_chat = model.start_chat(history=[])

        # Create a selectbox for topics
        selected_topic = st.selectbox("Select a topic:", list(self.topics.keys()))
        if selected_topic:
            self.show_subtopics(selected_topic)

    def show_subtopics(self, selected_topic):
        st.subheader(selected_topic)

        # Generate subtopics using the model
        prompt = f"List subtopics for {selected_topic}."
        subtopics_response = model.generate_content(prompt)
        subtopics = subtopics_response.text.strip().split('\n')

        # Display subtopics
        st.write("\n".join(subtopics))

        # Initialize chat history if not already present in session
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []

        # Display chat history
        for user_question, gemini_answer in st.session_state.chat_history:
            st.write(f"**You:** {user_question}")
            st.write(f"**Gemini:** {gemini_answer}")

        # Query box and button
        query = st.text_area("Ask a question:")
        if st.button("Ask"):
            self.ask_query(query, selected_topic)

    def ask_query(self, query, selected_topic):
        if query:
            # Process the query and pass it through the chat
            self.generate_response(query, selected_topic)

    def generate_response(self, question, selected_topic):
        try:
            # Sending the question to the ongoing chat session
            chat = st.session_state.gemini_chat
            prompt = f"Answer the following question related to {selected_topic}: {question}"
            response = chat.send_message(prompt)
            answer = response.text.strip()

            # Update chat history
            st.session_state.chat_history.append((question, answer))

            # Rerun Streamlit to refresh UI with updated chat history
            st.experimental_rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Run the application
if __name__ == "__main__":
    TutorApp()
