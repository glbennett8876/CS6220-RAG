# import subprocess
# import sys

# required_packages = [
#     "streamlit",
#     "langchain_community",
#     "langchain.tools",
#     "langchain.agents",
#     "langchain_openai",
#     "langchain_chroma",
#     "jq",
#     "llama_index"
# ]

# def install_and_import_packages(packages):
#     for package in packages:
#         try:
#             __import__(package)
#         except ImportError:
#             print(f"Package {package} not found. Installing...")
#             subprocess.check_call([sys.executable, "-m", "pip", "install", package])
#         else:
#             print(f"Package {package} is already installed.")

# install_and_import_packages(required_packages)

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import tempfile
from langchain_community.document_loaders import JSONLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_json_chat_agent
import json
from old_model import prompt
from model_base import Model
from medical_model_base import Medical_Model
from Display import Display

print("All required packages are installed and imported successfully.")



def implement_langchain(uploaded_files):
    # model = Model()
    model = Medical_Model()
    model2 = Model()
    data = process_files(model, uploaded_files)

    if data:
        question = st.form(key='my_form', clear_on_submit=True)
        model.question = question.text_input(label="Enter a question")
        submit_button = question.form_submit_button("Submit request")
        display = Display(model.question, "Json RAG with Custom Prompt")

        if model.question:
            run_rag(model, display, data)
            run_agent(model, data)
        else:
            display.container.empty()

def process_files(model, uploaded_files):
    selected_files = [None] * len(uploaded_files)
    st.write("----------\nSelect Files \n ----------")
    for i in range(len(uploaded_files)):
        button = st.checkbox(f"{uploaded_files[i].name}")
        selected_files[i] = button
    selected_files = [uploaded_files[i] for i, selected in enumerate(selected_files) if selected]
    st.write("----------")

    json_data = []
    for i, file in enumerate(selected_files):
        if file.name.endswith('.json'):
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_file:
                temp_file.write(file.getvalue())
                temp_file_path = temp_file.name

            json_file = json.load(file)

            def metadata_func(record: dict, metadata: dict) -> dict:
                metadata["name"] = record.get("name")
                return metadata
        
            loader = JSONLoader(
                file_path=temp_file.name,
                jq_schema = '.',
                text_content=False,
                metadata_func=metadata_func
            )

            docs = loader.load()
            for doc in docs:
                doc.metadata['file'] = file.name
            
            json_data.extend(docs)
            model.add_route(file.name, docs)

        else:
            st.write("Invalid file type")
            break
    return json_data


def run_rag(model, display, data):
    display.container.write("Processing...")

    # Base retrieval
    model.get_retriever(data)
    chunks = model.retriever.invoke(model.question)
    
    retriever_data = []
    for i in range(len(chunks)):
        chunk = chunks[i].page_content.replace("\n", " ")
        retriever_data.append(f'Chunk {i+1}:')
        retriever_data.append(chunk)
    
    #display.dic['Base retrieval'] = "\n\n".join(retriever_data)

    # RAG with hub prompt
    #display.dic['RAG with hub prompt'] = model.create_rag_chain(custom_prompt=False)

    # RAG with custom prompt
    display.dic['Question'] = model.create_rag_chain()
    display.show()
    # Semantic router guard
    #if button := st.button('Semantic Router'):
    #    display.dic['Semantic Router Guard'] = model.semantic_router()

    #display.show()


def run_agent(model, data):
    st.header("Agent")

    model.get_retriever(data)
    tool = create_retriever_tool(
        retriever = model.retriever,
        name = "json_retriever",
        description = "Searches and returns relevant data from a variety of input jsons to answer questions",
    )

    tools = [tool]
    agent = create_json_chat_agent(model.llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )

    if model.question:
        result = agent_executor.invoke({"input": model.question})
        st.markdown(f'Question: {result["input"]}')
        st.markdown(f'Answer: {result["output"]}')