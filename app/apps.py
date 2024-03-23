from dataclasses import dataclass
import streamlit as st

from langchain_community.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationChain
from langchain.prompts.prompt import PromptTemplate
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

from prompts.prompts import Template
from typing import Literal
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import NLTKTextSplitter
from PyPDF2 import PdfReader

@dataclass
class Message:
    """Class for keeping track of interview history."""
    origin: Literal["human", "ai"]
    message: str

def save_vector(resume):
    """embeddings"""

    pdf_reader = PdfReader(resume)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    # Split the document into chunks
    text_splitter = NLTKTextSplitter()
    texts = text_splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)
    return docsearch

def initialize_session_state_resume(resume):
    # convert resume to embeddings
    if 'docsearch' not in st.session_state:
        st.session_state.docsearch = save_vector(resume)
    # retriever for resume screen
    if 'retriever' not in st.session_state:
        st.session_state.retriever = st.session_state.docsearch.as_retriever(search_type="similarity")
   
    if "resume_history" not in st.session_state:
        st.session_state.resume_history = []
        st.session_state.resume_history.append(Message(origin="ai", message="Hello, I am your interivewer today. I will ask you some questions regarding your resume and your experience. Please start by saying hello or introducing yourself. Note: The maximum length of your answer is 4097 tokens!"))
    # token count
    if "token_count" not in st.session_state:
        st.session_state.token_count = 0
    # memory buffer for resume screen
    if "resume_memory" not in st.session_state:
        st.session_state.resume_memory = ConversationBufferMemory(human_prefix = "Candidate: ", ai_prefix = "Interviewer")
    #guideline for resume screen
    if "resume_guideline" not in st.session_state:
        llm = ChatOpenAI(
        model_name = "gpt-3.5-turbo",
        temperature = 0.5,)
        st.session_state.resume_guideline = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type_kwargs={},
            chain_type='stuff',
            retriever=st.session_state.retriever, 
            memory = st.session_state.resume_memory).run("Create an interview guideline and prepare only two questions for each topic. Make sure the questions tests the knowledge")
    # llm chain for resume screen
    if "resume_screen" not in st.session_state:
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
        st.session_state.resume_screen =  ConversationChain(prompt=PROMPT, llm = llm, memory = st.session_state.resume_memory)


# st.sidebar.success("Select a demo above.")
st.set_page_config(page_title="Test YOU")

def main_page():
    st.title("AI Interviewer")
    resumes = st.file_uploader("Upload your resume")


    if resumes is not None:
        initialize_session_state_resume(resumes)

        for message in st.session_state.resume_history:
            with st.chat_message(message.origin):
                st.markdown(message.message)

        if user_input := st.chat_input("Chat with me!"):
            # Display user message in chat message container
            with st.chat_message("human"):
                st.markdown(user_input)
            
            # Add user message to chat history
            st.session_state.resume_history.append(Message(origin="human", message=user_input))
            st.session_state.token_count += len(user_input.split())


            # Generate bot response
            bot_response = st.session_state.resume_screen.invoke(user_input)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.markdown(f"Bot: {bot_response}")

            # Add assistant response to chat history
            st.session_state.resume_history.append(Message(origin="ai", message=bot_response))
        

if __name__ == "__main__":
    main_page()