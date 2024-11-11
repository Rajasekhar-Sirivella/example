import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
import pandas as pd
from dotenv import load_dotenv
import os, json
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
os.environ['GROQ_API_KEY']=os.getenv('GROQ_API_KEY')
google_api_key=os.getenv("GEMINI_API_KEY")

examples=[{"system": """[
           {"company-name":"TCS",
           "short-term-assets":14780,
           "long-term-assets":1589,
           "short-term-liabilities":23750,
           "long-term-liabilities":5552,
           "equity":78234
           },
           {"company-name":"Accenture",
           "short-term-assets":56321,
           "long-term-assets":8888,
           "short-term-liabilities":98000,
           "long-term-liabilities":3422,
           "equity":67544
           }
           ]"""
        }]

example_prompt = ChatPromptTemplate.from_messages(
    [
        ("ai", "{system}"),
    ]
)

#Define a few-shot prompt template
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

#Final prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at analyzing Tables, Annual Reports and pdfs. provide the output in JSON format without any additional explanation, notes, or text. Ensure that the response is strictly valid JSON. The response contains only 2023 year annual report data."),
    few_shot_prompt,
    ("human", "Analyze the provided docs and provide response in the form of json format and it contains company-name, short-term-assets, long-term-assets, short-term-liabilities, long-term-liabilities, and equity. The Docs are {docs}"),
])

if __name__=="__main__":

    st.set_page_config(page_title="Anomaly detector", page_icon="🦜")
    st.title("🦜 Anomaly detector")
    st.subheader('Summarize PDF Document')
    
    groq_api_key=os.getenv("GROQ_API_KEY")
    #llm=ChatGroq(groq_api_key=groq_api_key,model_name="Llama3-8b-8192")
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=google_api_key)
    output_parser=JsonOutputParser()
    # prompt_template="""Provide a summary of the short term assets, long term assets, short term liabilities and long term liabilities with amount
    # from Consolidated balance sheets or Consolidated statements of financial position of 2023 year from Content 
    # and provide a summary of company in one line at starting.
    # Content : {text}"""
    # prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

    uploaded_files=st.file_uploader("Choose A PDf file", type="pdf", accept_multiple_files=True)
    df=pd.DataFrame()
    if uploaded_files:
        for upload_file in uploaded_files:
            temppdf=f"./temp.pdf"
            with open(temppdf,'wb') as file:
                file.write(upload_file.getvalue())
                file_name=upload_file.name
            try:
                loader=PyPDFLoader(temppdf)
                docs=loader.load()
                chain = prompt | llm | output_parser
                # chain=load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt)
                # output_summary=chain.run(docs)
                output_summary = chain.invoke({"docs": docs})[0]
                df=df._append(output_summary, ignore_index = True)
                st.success(file_name)
                st.write(output_summary)
            except Exception as e:
                st.exception(f"Exception;{e}")
        st.dataframe(df)
        st.success("Line Chart : ")
        st.line_chart(df, x="company-name", y=["short-term-assets", "long-term-assets", "short-term-liabilities", "long-term-liabilities", "equity"], x_label="Names of Companies", y_label="Assets and Liabilities")
        st.scatter_chart(df,x="company-name", y=["short-term-assets", "long-term-assets", "short-term-liabilities", "long-term-liabilities", "equity"], x_label="Names of Companies", y_label="Assets and Liabilities")


