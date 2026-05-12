from langchain_google_genai import GoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader,CSVLoader
 
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_core.prompts import PromptTemplate           


from langchain_community.document_loaders import DirectoryLoader  
import streamlit as st

 
embeddings = HuggingFaceEmbeddings( 
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


import os    
from dotenv import load_dotenv         



load_dotenv()      



docs = []

# PDF   




loader=PyPDFLoader("Free-English-Grammar-eBook-Beginner.pdf")


st.success("Chat English Grammar Book")




doc=loader.load()                             

splitter=RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200

)



if os.path.exists("provided_index"):

   
    vectorstores=FAISS.load_local("provided_index",embeddings, allow_dangerous_deserialization=True)  
    print("✅ Loaded existing data index")
else:
    chunks=splitter.split_documents(doc) 
    vectorstores=FAISS.from_documents(chunks,embeddings)   

    vectorstores.save_local("provided_index")
retrivers=vectorstores.as_retriever(
    search_kwargs={'k':2}
)  
st.write("Data Loaded ")




  
             
   

client=GoogleGenerativeAI(
    model="gemini-3-flash-preview",  
    api_key="AIzaSyC_r7W2jri1wJVN9YvfbjQYDjUPUwkzabE"        
   
    
)         


 


prompt = PromptTemplate(
    template="""
    Answer the query using the provided data and in the way you dont find the ans of query in provided then use your own knowledge to give 
    me suitable ans


    Query: {query}
    Data: {context} 
    history:{history}

    Format the answer clearly.   
    """,
    input_variables=["query", "context","history"] 
)      




      
    




if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if "chat" not in st.session_state:
    st.session_state.chat = []

if "summary" not in st.session_state:
    st.session_state.summary = ""



if not st.session_state.authenticated==False:

    password = st.text_input(
        "Enter Password",
        type="password",
        key="password_input"
    )

    if st.button("Login"):

        if password == "1234":

            st.session_state.authenticated = True
            st.success("Access Granted")

        else:

            st.error("Wrong Password")






else:

    st.title("AI Chatbot")

    query = st.text_input(
        "Ask me",
        key="query_input"
    )

    if st.button("Send"):

        try:

            st.write("Step 1: Button clicked")

            if query.strip() == "":

                st.warning("Enter query")
                st.stop()

            # ==================================
            # RETRIEVER
            # ==================================

            st.write("Step 2: Retrieving docs")

            docsfile = retrivers.invoke(query)

            st.write("Retriever Working")

            # ==================================
            # FORMAT DOCS
            # ==================================

            def format_docs(docs):

                return "\n\n".join(
                    doc.page_content
                    for doc in docs
                )

            context = format_docs(docsfile)

            st.write("Step 3: Context created")

            # ==================================
            # HISTORY
            # ==================================

            history_text = (
                st.session_state.summary
                + "\n"
                + "\n".join(
                    st.session_state.chat
                )
            )

            st.write("Step 4: History ready")

            # ==================================
            # CHAIN
            # ==================================

            val = prompt | client

            st.write("Step 5: Chain created")

            ans = val.invoke({
                "query": query,
                "context": context,
                "history": history_text
            })

            st.write("Step 6: Model response received")

            st.success(ans)

            # ==================================
            # STORE CHAT
            # ==================================

            st.session_state.chat.append(
                f"User: {query}"
            )

            st.session_state.chat.append(
                f"Bot: {ans}"
            )

        except Exception as e:

            st.error(f"ERROR: {e}")




    
 
 

    

    












    

    







