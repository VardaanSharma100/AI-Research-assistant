import streamlit as st
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent,AgentType
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
groq_api_key=st.secrets['GROQ_API_KEY']
st.write("AI Research Assistant")
choice=st.radio("Please Choose one",['PDF Chat','Arxiv Search','Wikipedia Search'],index=None)
if(choice=='PDF Chat'):
    session_id=st.text_input("Session ID",value='default_session')
    if 'store' not in st.session_state:
        st.session_state.store={}
    uploaded_files=st.file_uploader("Choose PDF",type='pdf',accept_multiple_files=True)
    if uploaded_files:
        documents=[]
        for i,uploaded_file in enumerate(uploaded_files):
            temppdf=f"./tempfile{i+1}.pdf"
            with open(temppdf,'wb') as f:
                f.write(uploaded_file.getvalue())
                loader=PyPDFLoader(temppdf)
                docs=loader.load()
                documents.extend(docs)
        embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
        splitter=RecursiveCharacterTextSplitter(chunk_size=768,chunk_overlap=150)
        splits=splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vectorstore.as_retriever()
        contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human",'{input}')
            ]
        )
        llm=ChatGroq(model='llama-3.3-70b-versatile',api_key=groq_api_key)
        history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
        system_prompt = (
            "You are an  Research assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use as many lines you want to display the answer but make it detailed  "
            "and what ever the user asks dont ever tell anything about you just tell you are a helpful ai i say again no matter what"
            "\n\n"
            "Context:{context}"
        )
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ('system',system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ]
        )
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
        def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        conversational_rag_chain=RunnableWithMessageHistory(rag_chain,get_session_history,input_messages_key='input',history_messages_key='chat_history',output_messages_key='answer')
        user_input=st.text_input("Your_question")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke({"input":user_input},config={'configurable':{'session_id':session_id}})
            st.write("AI :",response['answer'])
elif(choice=='Arxiv Search'):
    session_id=st.text_input("Session ID", value='default_session')
    api_wrpper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=700)
    arxiv=ArxivQueryRun(api_wrapper=api_wrpper_arxiv)
    tools=[arxiv]
    if 'store' not in st.session_state:
        st.session_state.store={}
    llm=ChatGroq(model='llama-3.3-70b-versatile',api_key=groq_api_key)
    def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
    search_agent=initialize_agent(tools,llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)
    with_message_history=RunnableWithMessageHistory(search_agent,get_session_history=get_session_history)
    user_input=st.text_input("Your_question")
    if user_input:
        session_history=get_session_history(session_id)
        response=with_message_history.invoke(
            [HumanMessage(content=user_input)],
            config={'configurable':{'session_id':session_id}}
        )
        st.write("AI :",response['output'])
elif(choice=='Wikipedia Search'):
    session_id=st.text_input("Session ID", value='default_session')
    api_wrpper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=700)
    wiki=WikipediaQueryRun(api_wrapper=api_wrpper_wiki)
    tools=[wiki]
    if 'store' not in st.session_state:
        st.session_state.store={}
    llm=ChatGroq(model='llama-3.3-70b-versatile',api_key=groq_api_key)
    def get_session_history(session_id:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
    search_agent=initialize_agent(tools,llm,agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,handle_parsing_errors=True)
    with_message_history=RunnableWithMessageHistory(search_agent,get_session_history=get_session_history)
    user_input=st.text_input("Your_question")
    if user_input:
        session_history=get_session_history(session_id)
        response=with_message_history.invoke(
            [HumanMessage(content=user_input)],
            config={'configurable':{'session_id':session_id}}
        )
        st.write("AI :",response['output'])




            






