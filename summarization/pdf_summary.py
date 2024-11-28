import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import ChatBedrock

def extract_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()  
    return text


def summarize_runbook(runbook_text):
    # Split the text into smaller chunks (if it's long)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=50
    )
    # Split runbook text into Document objects
    split_docs = text_splitter.create_documents([runbook_text])
    
    # Initial concise summary prompt template
    prompt_template = """Write a concise summary of the following:
    {text}
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Refine template for further summarization
    refine_template = (
        "Your job is to produce a final summary.\n"
        "provide summary in bullet points and below format.\n"
        " 1. Problem statement"
    "2. POV(solution apparoch)"
    "3. any other important details"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary."
        "If the context isn't useful, return the original summary."
    )
    refine_prompt = PromptTemplate.from_template(refine_template)
    
    # Initialize LLM (LangChain with Bedrock)
    llm = ChatBedrock(
        credentials_profile_name="default",
        #model_id="anthropic.claude-3-haiku-20240307-v1:0",
        model_id="amazon.titan-text-express-v1",
        #model_id="meta.llama3-8b-instruct-v1:0",
        model_kwargs={
        "temperature": 0.1,
        "topP": 0.9})

# Load the refine chain
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=prompt,
        refine_prompt=refine_prompt,
        return_intermediate_steps=True,
        input_key="input_documents",
        output_key="output_text",
        verbose=True
    )

    # Run the chain
    result = chain({"input_documents": split_docs}, return_only_outputs=True)

    return result['output_text']


# Streamlit session state
if 'runbook_text' not in st.session_state:
    st.session_state.runbook_text = None
if 'summary' not in st.session_state:
    st.session_state.summary = None

st.title("Summarizer")

# File upload
uploaded_file = st.file_uploader("Upload your Runbook (PDF)", type="pdf")

if uploaded_file is not None:
    with st.spinner("Extracting text from the PDF..."):
        runbook_text = extract_from_pdf(uploaded_file)
        st.session_state.runbook_text = runbook_text  # Store extracted text in session state

# Display extracted text if available
if st.session_state.runbook_text:
    st.write("### Extracted Text:")
    st.write(st.session_state.runbook_text[:500])  # Display first 500 characters as a preview

# Summarize button
if st.session_state.runbook_text and st.button("Summarize Runbook"):
    with st.spinner("Summarizing the runbook..."):
        summary = summarize_runbook(st.session_state.runbook_text)
        st.session_state.summary = summary  # Store summary in session state

# Display summary if available
if st.session_state.summary:
    st.write("### Summary:")
    st.write(st.session_state.summary)
