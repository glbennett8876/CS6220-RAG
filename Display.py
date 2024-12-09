import streamlit as st

class Display:
    def __init__(self, question, split):
        self.container = st.empty()
        self.question = question
        self.split = split
        self.dic = {}

    def show(self):
        self.container.empty()
        
        formatted_dict = f'## {self.split}\n\n'
        
        for key, value in self.dic.items():
            formatted_dict += f"{key}: {self.question}\n\n{value}\n\n----------\n\n"

        self.container.markdown(formatted_dict)