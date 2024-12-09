import streamlit as st

import demo

def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "RAG!PASSWORD!2024":
            st.session_state['logged_in'] = True
            st.session_state.langchain = False
            st.session_state.llama_index = False
            st.rerun()
        else:
            st.error("Invalid username or password")

def main():
    st.title("Medical Question Answering System")

    
    hospital_data = demo.main()
    st.write("--------\n Select Hospital")
    selected_hospital = hospital_data['hospital'].iloc[0]
    st.write(f"Analyzing data for hospital: {selected_hospital}")
    # Prepare JSON data for the selected hospital
    json_data = demo.prepare_json_data(hospital_data)
    # Call the implement function with the prepared JSON data
    demo.implement(json_data, selected_hospital)


st.set_page_config(
    page_title="Retrieval Augmented Generation",
    layout="wide",
)

if st.session_state.get('logged_in', False):
    main()
else:
    login()
