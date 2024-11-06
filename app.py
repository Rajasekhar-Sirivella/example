from langchain_google_genai import GoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_aws import ChatBedrock
import os, json
from dotenv import load_dotenv
import streamlit as st
from io import StringIO
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()

google_api_key=os.getenv("GEMINI_API_KEY")
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

def gemini_translate(source_db, target_db, sql):
    try:
        llm=GoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key, temperature=0.1)
        #prompt = """Translate the following SQL query from {source_db} to {target_db}:\n\nSQL Query:\n{sql}"""
        #prompt="""Translate this function from {source_db} into {target_db}. function: {sql}"""
        prompt = """
        Translate the following SQL query from {source_db} to {target_db} SQL syntax. If the {source_db} query contains joins within a DELETE or UPDATE statement, please rewrite the logic using {target_db}-compatible syntax, such as using subqueries or EXISTS clauses.

        {source_db} Query:
        {sql}

        {target_db} SQL Query:
        """
        final_prompt = prompt.format(source_db=source_db, target_db=target_db, sql=sql)
        # output_parser=JsonOutputParser()
        # chain=llm | output_parser
        response = llm.invoke(final_prompt)
        return response
    except Exception as ex:
        print(ex)

def groq_translate(source_db, target_db, sql):
    try:
        llm=ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant")
        prompt = """Translate the following SQL query from {source_db} to {target_db}:\n\nSQL Query:\n{sql}."""
        final_prompt = prompt.format(source_db=source_db, target_db=target_db, sql=sql)
        # output_parser=JsonOutputParser()
        # chain=llm | output_parser
        response = llm.invoke(final_prompt)
        return response
    except Exception as ex:
        print(ex)
def ollana_translate(source_db, target_db, sql):
    try:
        llm=OllamaLLM(model="codellama", temperature=0.1)
        prompt = """Translate the following SQL query from {source_db} to {target_db}:\n\nSQL Query:\n{sql}"""
        final_prompt = prompt.format(source_db=source_db, target_db=target_db, sql=sql)
        # output_parser=JsonOutputParser()
        # chain=llm | output_parser
        response = llm.invoke(final_prompt)
        return response
    except Exception as ex:
        print(ex)

def get_output(model, source_database, target_database, input_text):
    if model=="gemini-1.5-flash":
        resp=gemini_translate(source_database, target_database, input_text)
        st.write(resp)
    elif model=="llama-3.1-8b-instant":
        resp=groq_translate(source_database, target_database, input_text)
        st.write(resp)
    elif model=="codellama":
        resp=ollana_translate(source_database, target_database, input_text)
        st.write(resp)

st.set_page_config(
    page_title=' A SQL Generative Pre-trained Transformer',
    layout='wide',
    initial_sidebar_state='expanded'
)

databases = ['Oracle', 'SQLServer', 'MySQL', 'DB2', 'PostgreSQL', 'Snowflake', 'Redshift']
models = ['gemini-1.5-flash', 'llama-3.1-8b-instant', 'codellama']
st.sidebar.header('A SQL Transformer for Migration')
model = st.sidebar.selectbox(label='Model', options=models, index=0)

source_database = st.sidebar.selectbox(
    label='Source Database',
    options=databases,
    index=0
)

target_database = st.sidebar.selectbox(
    label='Target Database',
    options=databases,
    index=4
)

input_text = st.sidebar.text_area(
    label='Insert SQL Query',
    height=200,
    placeholder='select id from customer where rownum <= 100'
)

input_file = st.sidebar.file_uploader("Upload your SQL file here",accept_multiple_files=False)

if input_text:
    get_output(model, source_database, target_database, input_text)
else:
    if input_file is not None:
        stringio = StringIO(input_file.getvalue().decode("utf-8"))
        sql = stringio.read()
        get_output(model, source_database, target_database, sql)