import json
from typing import Literal, Union, List
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.chat_models import ChatOpenAI

from dataclasses import dataclass
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")

@dataclass
class Message:
    '''dataclass for keeping track of the messages'''
    origin: Literal["human", "ai"]
    message: str

@dataclass
class Question:
    question: str
    type: Literal["personal", "behavioral", "situational"]


@dataclass
class Evaluation:
    evaluation: Literal["good", "average", "bad"]
    feedback: str | None
    reason: str | None
    samples: list[str] | None


class QuestionGeneratorAgent:
    def __init__(self) -> None:
        self.client = OpenAI(api_key=openai_key)
        self.system_prompt = """You are a non-technical interviewer that interviews \
across the following categories:
- personal
- behavioral
- situational

You will be provided with a candidate's description.

Generate {n_questions} questions, ensuring that there is a question for each category \
and the questions should be based on the candidate's description.

* You answer strictly as a list of JSON objects. Don't include any other verbose texts, \
and don't include the markdown syntax anywhere.

JSON format:
[
    {{"question": "<personal_question>", "type": "personal"}},
    {{"question": "<behavioral_question>", "type": "behavioral"}},
    {{"question": "<situational_question>", "type": "situational"}},
    ...more questions to make up {n_questions} questions
]"""

        self.user_prompt = "Candidate Description:\n{description}"

    def __call__(self, description: str, n_questions: int = 3) -> List[Question] | None:
        return self.generate_questions(description, n_questions)

    def run(self, description: str, n_questions: int = 3) -> List[Question] | None:
        return self.generate_questions(description, n_questions)

    def generate_questions(self, description: str, n_questions: int = 3) -> Union[List[Question], None]:
        try:
            # Ensure that there are at least 3 questions
            if n_questions < 3:
                n_questions = 3

            output = self.client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content": self.system_prompt.format(n_questions=n_questions),
                    },
                    {
                        "role": "user",
                        "content": self.user_prompt.format(description=description),
                    },
                ],
                temperature=0.5,
                max_tokens=1024,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
            )

            if output.choices[0].finish_reason != "stop":
                return None

            if not output.choices[0].message.content:
                return None

            questions = [Question(**q) for q in json.loads(output.choices[0].message.content)]
            return questions
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        

"""                     STREAMLIT BACKEND                """


st.title("Ready to test your soft skills")

question_generator = QuestionGeneratorAgent()


# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
if user_input := st.chat_input("Chat with me!"):
    # Display user message in chat message container
    with st.chat_message("human"):
        st.markdown(user_input)
    
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Generate interview questions based on user input
    questions = question_generator.run(user_input, 4)

    # Display generated questions and collect user responses
    if questions:
        for idx, q in enumerate(questions):
            # Display question
            with st.empty():
                st.write(f"Bot: {q.question}")

            # User input for answering the question
            user_response = st.text_input(f"Your answer to question {idx + 1}:", key=f"response_{idx}")

            # Add user response to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_response})

            # Add bot question to chat history
            st.session_state.chat_history.append({"role": "assistant", "content": f"Bot: {q.question}"})


# # User input
# user_input = st.text_area(label="Provide a brief description here")
# n_questions = st.number_input(label='Preffered number of questions', min_value=1, max_value=10)
# button = st.button('Submit')

# # Initialize chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # User input
# if button:
#     # Display user message in chat message container
    
    
#     # Add user message to chat history
#     # st.session_state.chat_history.append({"role": "user", "content": user_input})
#     # st.session_state.token_count += len(user_input.split())  # Uncomment if you need token count

#     # Generate interview questions based on user input
#     questions = question_generator.run(user_input, n_questions)

#     # Display generated questions and collect user responses
#     if questions:
#         for idx, q in enumerate(questions):
#             # Display question
#             with st.chat_message("assistant"):
#                 st.markdown(f"Bot: {q.question}")
            

#             # User input for answering the question
#             user_response = st.text_input(f"Your answer to question {idx + 1}:", key=f"response_{idx}")

#             # Add user response to chat history
#             st.session_state.chat_history.append({"role": "user", "content": user_response})
#             with st.chat_message("human"):
#                 st.markdown(f"Bot: {user_input}")

#             # Add bot question to chat history
#             st.session_state.chat_history.append({"role": "assistant", "content": f"Bot: {q.question}"})










# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])



# if button:
#     # Generate interview questions based on user input
#     questions = question_generator.run(user_input, n_questions)

#     # Display generated questions
#     if questions:
#         for q in questions:
#             with st.chat_message("assistant"):
#                 st.markdown(f"Bot: {q.question}")
#             # User input for answering the question
#             user_response = st.text_input("Your answer:")

#             # Add user response to chat history
#             st.session_state.chat_history.append({"role": "user", "content": user_response})                
#             # Add assistant response to chat history
#             st.session_state.chat_history.append({"role": "assistant", "content": f"Bot: {q.question}"})

#     # Add user input to chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_input})










# import streamlit as st
# from behave import QuestionGeneratorAgent,ResponseEvaluationAgent
# import os

# st.title("Ready to test your soft skills")


# # Instantiate the QuestionGeneratorAgent
# openai_key = os.getenv("OPENAI_API_KEY")

# question_generator = QuestionGeneratorAgent()

# # Initialize chat history
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []

# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# # User input
# user_input = st.text_area(label="Provide a brief description here")
# n_questions = st.number_input(label='Preffered number of questions',min_value=1,max_value=10)
# button = st.button('Submit')

# if button:
#     # Generate interview questions based on user input
#     questions = question_generator.run(user_input, n_questions)

#     # Display generated questions
#     if questions:
#         for q in questions:
#             with st.chat_message("assistant"):
#                 st.markdown(f"Bot: {q.question}")   
#             # Add assistant response to chat history
#             st.session_state.chat_history.append({"role": "assistant", "content": f"Bot: {q.question}"})

#     # Add user input to chat history
#     st.session_state.chat_history.append({"role": "user", "content": user_input})


