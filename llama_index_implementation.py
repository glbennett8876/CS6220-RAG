from llama_index.core.indices.struct_store import JSONQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent.legacy.react.base import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import ToolMetadata
import streamlit as st
import json
import tempfile
import os

os.environ["OPENAI_API_KEY"] = ""

llm = OpenAI(model_name="gpt-3.5-turbo")

def process_files(uploaded_files):
    selected_files = [None] * len(uploaded_files)
    st.write("----------\nSelect Files \n ----------")
    for i in range(len(uploaded_files)):
        button = st.checkbox(f"{uploaded_files[i].name}")
        selected_files[i] = button
    selected_files = [uploaded_files[i] for i, selected in enumerate(selected_files) if selected]
    st.write("----------")

    tools = []
    for i, file in enumerate(selected_files):
        if file.name.endswith('.json'):
            json_file = json.load(file)
            
            json_engine = JSONQueryEngine(
                json_value=json_file,
                json_schema={},
                name=f"{file.name[:-5]}",
            )
            
            tool = QueryEngineTool(query_engine=json_engine, 
                metadata=ToolMetadata(
                    name=f"{file.name[:-5]}",
                    description=f"This contains all the information about the manager including his email"
                )  
            )
            tools.append(tool)
            

        else:
            st.write("Invalid file type")
            break
    return tools

def query(tools):
    if not tools: 
        return
    context = """Purpose: The primary role of this agent is to retrieve accurate data from a collection of jsons
            in order to answer use queries."""
    
    agent = ReActAgent.from_tools(tools, llm=llm, verbose=True, context=context)
    question_form = st.form(key='my_form', clear_on_submit=True)
    question = question_form.text_input(label="Enter a question")
    submit_button = question_form.form_submit_button("Submit request")
    if question:
        result = agent.chat(question)
        st.write(f"Question: {question}")
        st.write(f'Answer: {str(result)}')



