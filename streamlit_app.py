
#importing necessary libraries
from audioop import add
from pydoc import synopsis
from shlex import split
import streamlit as st
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
import os
from langchain.document_loaders import AsyncHtmlLoader
from langchain.document_transformers import BeautifulSoupTransformer
import re



# Vijay & his cousins will chase abrar and finally vijay will kill abrar on the airport after fight. 
#         Balbir will develop cancer and will die. 
#       His wife geetanjali will leave vijay for usa heartbroken vijay will hug his son for last time. 
    
#open ai API KEY
os.environ["OPENAI_API_KEY"]="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"
st.set_page_config(page_title="🦜🔗 🔫Kill Spoilers Demo Run")
st.title('🦜🔗 🔫 Kill Spoilers Demo Run')

#importing NLU embedding
embeddings=OpenAIEmbeddings()


#loading the vectorDb
synopsis_db=FAISS.load_local("synopsis_final_all", embeddings)
summary_db=FAISS.load_local("summary_final",embeddings)

#initlizing the vector db as retriver object
synopsis_retri=summary_db.as_retriever()
summary_retri=synopsis_db.as_retriever()

#this function splits the text array if any element is longer than 1000 it splits
def split_pipe(arr):
    #defining the text splitter
    text_splitter=RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap  = 50)
    empty_arr=[]
    #iterating through array
    for i in range(len(arr)):
        empty_arr.append(text_splitter.split_text(arr[i]))
    empty_arr_2=[]
    
    #reshaping into 1D array
    for i in range(len(empty_arr)):
        for j in range(len(empty_arr[i])):
            empty_arr_2.append(empty_arr[i][j])
                       
    return empty_arr_2
        

#defing the prompt template 
prompt_template = PromptTemplate.from_template(
"""
    You are AI assistant your task is to block spoilers of movies.
    
    For Given movie {summary} & {synopsis} answer whether given {review} is spoiler or not.
    
    Answer in binary format 1:Yes & 0:No,followed by Eplanation how its spoilers:
    """
)



#this helper function adds the data into vector db
def add_into_vectorDb(db,text):
    db.merge_from(FAISS.from_texts(text,embedding=OpenAIEmbeddings()))
    return db

#this reinitlize the vector db as retrever
def reinitialize_retriever(db):
    return db.as_retriever()


#helper function used to format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#this helper function is used to perform RAG and invoking the RAG chain
def generate_response(input_text):
   llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
   rag_chain = (
      {"summary": st.session_state.summary_retri | format_docs,"synopsis":st.session_state.synopsis_retri, "review": RunnablePassthrough()}
      |prompt_template|llm
      | StrOutputParser())
   st.info(rag_chain.invoke(input_text))


#this helper function uses beatiful soup for web scarpping and scraps synpsis of movie given movie id
def scrap_syno(movie_id):
    url=f"https://www.imdb.com/title/{movie_id}/plotsummary/?ref_=tt_stry_pl"
    html=AsyncHtmlLoader(url).load()
    docs_transformed = BeautifulSoupTransformer().transform_documents(html, tags_to_extract=["section"])
    pattern = re.compile(r'(?:Synopsis.+?){2}(.+)', re.IGNORECASE | re.DOTALL)
    text=pattern.search(docs_transformed[0].page_content).group(1).strip()
    pattern2=re.compile(r'^(.*?)(?=(?:Begin INLINE40|$))', re.IGNORECASE | re.DOTALL)
    return pattern2.search(text).group(1).strip()

#this helper function uses beatiful soup for web scarpping and scraps summary of movie given movie id
def scarp_summary(movie_id):
    url=f"https://www.imdb.com/title/{movie_id}/plotsummary/?ref_=tt_stry_pl"
    html=AsyncHtmlLoader(url).load()
    docs_transformed = BeautifulSoupTransformer().transform_documents(html, tags_to_extract=["section"])
    pattern = re.compile(r'(?:Summaries.+?){2}(.+)', re.IGNORECASE | re.DOTALL)
    text=pattern.search(docs_transformed[0].page_content).group(1).strip()
    pattern2=re.compile(r'^(.*?)(?=(?:Synopsis|$))', re.IGNORECASE | re.DOTALL)
    return pattern2.search(text).group(1).strip()

#function to add text into vector db



with st.form('my_form'):
    movie_id = st.text_area('Enter Imbd Movie ID:')
    add = st.form_submit_button('Submit movie ID')

    text = st.text_area('Enter review:', 'Enter reviews to identify as spoiler or not')
    submitted = st.form_submit_button('Submit')
    

    #its for state persistence such that object stays persistent with UI changes
    if 'synopsis_db' not in st.session_state:
        st.session_state.synopsis_db = FAISS.load_local("synopsis_final_all", embeddings)
    
    #its for state persistence such that object stays persistent with UI changes
    if 'summary_db' not in st.session_state:
        st.session_state.summary_db = FAISS.load_local("summary_final", embeddings)
    
    #its for state persistence such that object stays persistent with UI changes
    if 'summary_retri' not in st.session_state:
        st.session_state.summary_retri = summary_db.as_retriever()

    #its for state persistence such that object stays persistent with UI changes
    if 'synopsis_retri' not in st.session_state:
        st.session_state.synopsis_retri = synopsis_db.as_retriever()

    #if add is clicked then data is added into  vector db and reinitlized as retriver
    if add and (movie_id is not None):
        st.session_state.synopsis_db = add_into_vectorDb(st.session_state.synopsis_db, split_pipe([scrap_syno(movie_id)]))
        st.session_state.synopsis_retri=reinitialize_retriever(st.session_state.synopsis_db)
        
        st.session_state.summary_db = add_into_vectorDb(st.session_state.synopsis_db, split_pipe([scarp_summary(movie_id)]))
        st.session_state.summary_retri=reinitialize_retriever(st.session_state.summary_db)
        st.info('records added into  vecor db')
    #if submitted is clicked it then RAG chain is invoked
    if submitted:
        generate_response(text)
    


  

