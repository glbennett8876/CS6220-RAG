import os
import csv
import tempfile
import json
import pandas as pd
import streamlit as st
from langchain_community.document_loaders import JSONLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from medical_model import MedicalQA  # Import MedicalQA class
from model_base import Model

def main():
    current_dir = os.getcwd()
    csv_path = os.path.join(current_dir, 'filtered_admissions.csv')
    
    if os.path.exists(csv_path):
        admissions_data = pd.read_csv(csv_path)
    else:
        st.write("File 'filteredadmissions.csv' not found in the current directory.")
        return None

    hospitals = admissions_data['hospital'].unique()
    st.write("Hospitals found in the dataset:")
    st.write(hospitals)

    selected_hospital = st.selectbox("Select a hospital to analyze:", hospitals)

    if selected_hospital:
        hospital_data = admissions_data[admissions_data['hospital'] == selected_hospital]
        st.write(f"Data for {selected_hospital}:")
        st.write(hospital_data.head())
        return hospital_data
    return None

def prepare_json_data(hospital_data):
    json_data = []
    for _, row in hospital_data.iterrows():
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as temp_file:
            json.dump(row.to_dict(), temp_file, indent=2)
            temp_file_path = temp_file.name

        def metadata_func(record: dict, metadata: dict) -> dict:
            metadata["hospital"] = row["hospital"]
            return metadata

        loader = JSONLoader(
            file_path=temp_file.name,
            jq_schema='.',
            text_content=False,
            metadata_func=metadata_func
        )

        docs = loader.load()
        json_data.extend(docs)

    return json_data

def implement(json_data, selected_hospital):
    model = Model()

    # Initialize session states for storing results
    if "hospital_query_result" not in st.session_state:
        st.session_state["hospital_query_result"] = None

    if "medical_query_result" not in st.session_state:
        st.session_state["medical_query_result"] = None

    # Input for hospital query
    question = st.text_input(label=f"Enter a question about {selected_hospital} data:", key="hospital_query")
    submit_button = st.button("Submit Hospital Query")

    if submit_button and question:
        model.question = question
        st.write(f"Analyzing data for: {selected_hospital}")
        model.get_retriever(json_data)

        tool = create_retriever_tool(
            retriever=model.retriever,
            name="HOSPITAL",
            description=f"Searches and returns relevant data regarding {selected_hospital}.",
        )

        tools = [tool]
        human = """
        TOOLS
        Assistant can ask the user to use tools to look up information that may be helpful in answering the user's original question. The tools the human can use are:
        {tools} as well as chat history
        RESPONSE FORMAT INSTRUCTIONS
        ----------------------------
        When responding to me, please output a response in one of the formats below:
        Use this if you want the human is asking a question that has no relevance to any of the tools. 
        **Option #1** :
            "action": string, \ This action is not allowed. The action must be one of {tool_names}
            "action_input": string \ The action must be one of {tool_names}.

        **Option #2** :
        Use this if you want to respond directly to the human using the available tools.
            "action": "Final Answer",
            "action_input": string \ You should put what you want to return to use here

        USER'S INPUT
        --------------------
        Here is the user's input:
        {input}
        If you do not know the answer, say you do not have enough context.
        """ 
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a powerful assistant who provides answers to questions based on retrieved data using context and chat history"),
                ("human", human),
                MessagesPlaceholder("agent_scratchpad"),
                MessagesPlaceholder("chat_history"),
            ]
        )

        json_agent = create_json_chat_agent(model.llm, tools, prompt)
        agent_executor = AgentExecutor(agent=json_agent, tools=tools, handle_parsing_errors=True)
        agent_chain = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: model.memory,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        response = agent_chain.invoke({"input": model.question}, config={"configurable": {"session_id": "1"}})
        st.session_state["hospital_query_result"] = response["output"]
        st.write("Hospital Query Result:")
        st.write(st.session_state["hospital_query_result"])

    # Medical Query Section
    if st.session_state["hospital_query_result"]:
        medical_qa = MedicalQA(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")  # Ensure the correct model path
        medical_query = st.text_input(label="Ask a Medical Question", key="medical_query")
        medical_submit = st.button("Submit Medical Query")

        if medical_submit and medical_query:
            st.write("Using hospital data as context...")
            medical_context = st.session_state["hospital_query_result"]
            medical_answer = medical_qa.query(medical_query, context=medical_context)
            st.session_state["medical_query_result"] = medical_answer
            st.write("MedicalQA Result:")
            st.write(st.session_state["medical_query_result"])

    # Display results
    st.write("Previous Responses:")
    if st.session_state["hospital_query_result"]:
        st.write("Hospital Query Result (Preserved):")
        st.write(st.session_state["hospital_query_result"])

    if st.session_state["medical_query_result"]:
        st.write("MedicalQA Result (Preserved):")
        st.write(st.session_state["medical_query_result"])

if __name__ == "__main__":
    hospital_data = main()
    if hospital_data is not None:
        json_data = prepare_json_data(hospital_data)
        selected_hospital = hospital_data['hospital'].iloc[0]
        implement(json_data, selected_hospital)
