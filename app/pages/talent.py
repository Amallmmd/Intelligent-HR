import json
from typing import Literal, Union, List
from openai import OpenAI
from openai.types.chat.chat_completion import ChatCompletion
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts.prompt import PromptTemplate
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import NLTKTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dataclasses import dataclass
from dotenv import load_dotenv
from prompts.prompts import Template
import os
import streamlit as st
load_dotenv()
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

def embeddings(text: str):

    '''Create embeddings for the job description'''

    # nltk.download('punkt')
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)
    # Create emebeddings
    embeddings = OpenAIEmbeddings()
    textsearch = FAISS.from_texts(texts, embeddings)
    retriever = textsearch.as_retriever(search_tupe='similarity search')
    return retriever

def behaviour_test(text:str):# take arguments (short description and no. of questions to ask)
    # if n_questions < 3:
    #         n_questions = 3
    if "retriever" not in st.session_state:
        st.session_state.retriever = embeddings(text)

    if "b_history" not in st.session_state:
        st.session_state.b_history = []
        st.session_state.b_history.append(Message(origin="ai", message="I am your interviewer. I'll ask you some behavioral questions for evaluation. Let's start by Hello "))
    if "memory" not in st.session_state:
            st.session_state.memory = ConversationBufferMemory()
    if "guideline" not in st.session_state:
        llm = ChatOpenAI(model= "gpt-3.5-turbo",temperature=0.8)
        st.session_state.guideline = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type_kwargs={},
                chain_type='stuff',
                retriever=st.session_state.retriever, 
                memory = st.session_state.memory).run("Prepare the guidline ask 4 question to the candidate. make sure to check the behaviour of the candidate")
    if "conversation" not in st.session_state:
        llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.7, )
        PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template= """I want you to act as an interviewer strictly following the guideline in the current conversation.
            
                        Ask me questions and wait for my answers like a human. Do not write explanations.
                        Candidate has no assess to the guideline.
                        Only ask one question at a time. 
                        Do ask follow-up questions if you think it's necessary.
                        Do not ask the same question.
                        Do not repeat the question.
                        Candidate has no assess to the guideline.
                        You name is GPTInterviewer.
                        I want you to only reply as an interviewer.
                        Do not write all the conversation at once.
                        Candiate has no assess to the guideline.
                        
                        Current Conversation:
                        {history}
                        
                        Candidate: {input}
                        AI: """)
        st.session_state.conversation = ConversationChain(prompt=PROMPT, llm=llm,memory = st.session_state.memory)
        
        if "feedback" not in st.session_state:
            llm = ChatOpenAI(
            model_name = "gpt-3.5-turbo",
            temperature = 0.5,)
            st.session_state.feedback = ConversationChain(
                prompt=PromptTemplate(input_variables = ["history", "input"], template = Template.feedback_template),
                llm=llm,
                memory = st.session_state.memory,
            )
        
    # except Exception as e:
    #      pass
            

# def main():
#     st.title("Ready to test your soft skills")

#     if "input_text" not in st.session_state:
#         st.session_state.input_text = ""
#     if "b_history" not in st.session_state:
#         st.session_state.b_history = []

#     input_text = st.text_area(label="Provide a brief description here", value=st.session_state.input_text)
#     button = st.button("Submit")

#     if button:
#         st.session_state.input_text = input_text
#         behaviour_test(input_text)
#         for message in st.session_state.b_history:
#             with st.expander(message.origin):
#                 st.write(message.message)

#     if "conversation" in st.session_state:
#         if len(st.session_state.b_history) % 2 == 0:  # Bot's turn to speak
#             bot_input = ""
#             if len(st.session_state.b_history) > 0:
#                 bot_input = st.session_state.b_history[-1].message

#             bot_response = st.session_state.conversation.run(bot_input)

#             with st.expander("ai"):
#                 st.write(f"Bot: {bot_response}")
#             st.session_state.b_history.append(Message(origin="ai", message=bot_response))
#         else:  # User's turn to speak
#             user_input = st.text_input("Your response:")
#             if user_input:
#                 with st.expander("human"):
#                     st.write(user_input)

#                 st.session_state.b_history.append(Message(origin="human", message=user_input))

# if __name__ == "__main__":
#     main()




def main():
    st.title("Ready to test your soft skills")

    if "input_text" not in st.session_state:
        st.session_state.input_text = ""
    if "b_history" not in st.session_state:
        st.session_state.b_history = []

    input_text = st.text_area(label="Provide a brief description here", value=st.session_state.input_text)
    button = st.button("Submit")

    if button:
        st.session_state.input_text = input_text
        behaviour_test(input_text)
        for message in st.session_state.b_history:
            with st.expander(message.origin):
                st.write(message.message)

    if "conversation" in st.session_state:
        user_input = st.text_input("Chat:")
        if user_input:
            with st.expander("human"):
                st.write(user_input)

            st.session_state.b_history.append(Message(origin="human", message=user_input))
            bot_response = st.session_state.conversation.run(user_input)

            with st.chat_message("assistant"):
                st.markdown(f"Bot: {bot_response}")
            st.session_state.b_history.append(Message(origin="ai", message=bot_response))

if __name__ == "__main__":
    main()



# def main():
#     st.title("Ready to test your soft skills")

#     input_text = st.text_area(label="Provide a brief description here")
#     button = st.button("Submit")
#     if button:
#         behaviour_test(input_text)
#         for message in st.session_state.b_history:
#             with st.chat_message(message.origin):
#                 st.markdown(message.message)

#         if user_input:=st.chat_input("Chat!"):
#             with st.chat_message("human"):
#                 st.markdown(user_input)

#             st.session_state.b_history.append(Message(origin="human",message=user_input))
#             bot_response = st.session_state.conversation.run(user_input)

#             with st.chat_message("ai"):
#                 st.markdown(f"Bot:{bot_response}")
#             st.session_state.b_history.append(Message(origin="ai",message=bot_response))
# if __name__ =="__main__":
#     main()
    
