import streamlit as st
from pathlib import Path
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.utilities import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlalchemy import create_engine
import sqlite3
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import re
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.prebuilt import create_react_agent

load_dotenv()

# from langchain_openai import AzureChatOpenAI
from langchain_groq import ChatGroq
# from openai import AzureOpenAI
from audio_recorder_streamlit import audio_recorder

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

from langchain_google_vertexai.chat_models import ChatVertexAI
from google.cloud import aiplatform

PROJECT_ID = "tcs-cto-retail"
aiplatform.init(project=PROJECT_ID, location="us-central1")


# os.environ["AZURE_OPENAI_ENDPOINT"] = "https://documentsummary.openai.azure.com/"
# os.environ["AZURE_OPENAI_API_KEY"] = os.getenv("AZURE_OPENAI_API_KEY")

# # Whisper config
# azure_endpoint_whisper = "https://whisper-tts.openai.azure.com/"
# azure_key_whisper = os.getenv("WHISPER_OPENAI_API_KEY")

# Speech to text
# def transcribe_audio(audio_path):
#     client = AzureOpenAI(
#         api_key=azure_key_whisper,
#         api_version="2024-02-01",
#         azure_endpoint=azure_endpoint_whisper
#     )
#     with open(audio_path, "rb") as audio_file:
#         transcript = client.audio.transcriptions.create(
#             model="whisper",
#             file=audio_file,
#         )
#     return transcript.text

st.set_page_config(page_title="SQLDB", page_icon=":robot:")
st.title("Chat with any SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

# Radio options
radio_opt = ["Use SQLLite 3 DB-Products.db", "Connect to your SQL Database"]

select_opt = st.sidebar.radio(label="Choose your DB which you want to chat", options=radio_opt)

if radio_opt.index(select_opt) == 1:
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Enter MySQL Host")
    mysql_user = st.sidebar.text_input("Enter MySQL User")
    mysql_password = st.sidebar.text_input("Enter MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("Enter MySQL Database")
else:
    db_uri = LOCALDB

# Select the LLM accordingly
llm_choice=st.sidebar.selectbox("Choose the LLM", options=["VertexAI", "Groq", "Gemini"])
if llm_choice=="VertexAI":
    llm = ChatVertexAI(
        model="gemini-1.5-flash-002",
        temperature=0.4)
    st.sidebar.info("VertexAI LLM is selected")
elif llm_choice=="Groq":
    llm=ChatGroq(temperature=0.4, model_name="Llama3-8b-8192")
    st.sidebar.info("Groq LLM is selected")
elif llm_choice =="Gemini":
    llm = ChatGoogleGenerativeAI(temperature=0.5, model='gemini-1.5-pro')
    st.sidebar.info("Gemini LLM is selected")

if not db_uri:
    st.info("Please select a DB information and URI")

# Function to configure the database
@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    if db_uri == LOCALDB:
        dbfilepath = (Path(__file__).parent / "product.db").absolute()
        creator = lambda: sqlite3.connect(f"file:{dbfilepath}?mode=ro", uri=True)
        return SQLDatabase(create_engine("sqlite://", creator=creator))
    elif db_uri == MYSQL:
        if not (mysql_host and mysql_user and mysql_password and mysql_db):
            st.error("Please enter all MySQL connection details")
            st.stop()
        return SQLDatabase(create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}"))

if db_uri == MYSQL:
    db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
else:
    db = configure_db(db_uri)

SQL_PREFIX = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct SQLite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

To start you should ALWAYS look at the tables in the database to see what you can query.
Do NOT skip this step.
Then you should query the schema of the most relevant tables."""

system_message = SystemMessage(content=SQL_PREFIX)

## toolkit
toolkit=SQLDatabaseToolkit(db=db,llm=llm)
tools = toolkit.get_tools()

agent_executor = create_react_agent(llm, tools, state_modifier=system_message)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [{"role": "assistant", "query": "", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    if msg['role']=="user":
        st.chat_message(msg["role"]).write(msg["content"])
    elif msg['role']=='assistant' and msg["query"]!="":
        with st.chat_message("assistant"):
            st.write(msg['query'])
            st.write(msg['content'])
    else:
        st.chat_message(msg["role"]).write(msg["content"])

# Function to generate a dynamic plot
def plot_dynamic(data, plot_type):
    fig, ax = plt.subplots()
    
    if plot_type == "pie":
        ax.pie(data['values'], labels=data['labels'], autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
    elif plot_type == "bar":
        ax.bar(data['labels'], data['values'], color='blue')
        ax.set_xlabel('Labels')
        ax.set_ylabel('Values')
    elif plot_type == "line":
        ax.plot(data['labels'], data['values'], marker='o')
        ax.set_xlabel('Labels')
        ax.set_ylabel('Values')
    else:
        st.error("Unsupported plot type!")
        return
    
    st.pyplot(fig)

# Function to parse the response and generate a dynamic plot if needed
def handle_dynamic_response(response):
    if any(kw in response.lower() for kw in ["plot", "chart", "graph"]):
        # Identify plot type
        if "pie chart" in response.lower():
            plot_type = "pie"
        elif "bar chart" in response.lower():
            plot_type = "bar"
        elif "line chart" in response.lower():
            plot_type = "line"
        else:
            plot_type = None
        
        # Extract data
        matches = re.findall(r"([\w\s]+):\s*(\d+)", response)
        if matches:
            labels = [match[0].strip() for match in matches]
            values = [int(match[1]) for match in matches]
            data = {"labels": labels, "values": values}
            plot_dynamic(data, plot_type)
        else:
            st.error("Could not extract chart data from the response.")
    else:
        st.write(response)

# Record audio
# st.write("Record your query:")
# recorded_audio = audio_recorder()
# if recorded_audio:
#     st.write("Audio recorded successfully!")
#     audio_file = "recorded_audio.mp3"
#     with open(audio_file, "wb") as f:
#         f.write(recorded_audio)

#     transcribe_text = transcribe_audio(audio_file)
#     st.write("Transcribed Text:", transcribe_text)

#     # Append transcribed text to session state and process
#     if transcribe_text:
#         st.session_state.messages.append({"role": "user", "content": transcribe_text})
#         st.chat_message("user").write(transcribe_text)
#         with st.chat_message("assistant"):
#             streamlit_callback = StreamlitCallbackHandler(st.container())
#             response = agent.run(transcribe_text, callbacks=[streamlit_callback])

#             # Handle response, either plot or text
#             handle_dynamic_response(response)
#             st.session_state.messages.append({"role": "assistant", "content": response})

# Handle text input
user_query = st.chat_input(placeholder="Ask me anything from DB!")

if user_query:
    try:
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.chat_message("user").write(user_query)
        with st.chat_message("assistant"):
            response=agent_executor.invoke({"messages": [HumanMessage(content=user_query)]})
            #query=(response["messages"][-3].additional_kwargs['function_call']['arguments']) #returns string format
            query=response["messages"][-3].tool_calls[0]['args'] #returns json format
            output=response["messages"][-2].content
            # Handle response, either plot or text
            handle_dynamic_response(response["messages"][-2].content)
            result="Output : "+output+"\n\n\n"+response["messages"][-1].content #query response and llm response
            st.session_state.messages.append({"role":"assistant", "query": query,"content": result})
            st.write(query)
            st.write(result)
    except Exception as e:
        st.error("Error Occured... Try again.")
        print("The error is: ",e)
