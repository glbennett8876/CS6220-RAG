from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from semantic_router.encoders import OpenAIEncoder
from semantic_router import Route
from langchain_elasticsearch import ElasticsearchStore
from semantic_router.layer import RouteLayer
from langchain_openai import OpenAIEmbeddings
from elasticsearch import Elasticsearch
from langchain.agents import AgentExecutor, create_json_chat_agent
from langchain_community.document_loaders import JSONLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os 

#insert openapi key
os.environ['OPENAI_API_KEY'] = os.environ["OPENAI_API_KEY"] = ""
es = Elasticsearch('http://localhost:9200')
question = "What are examples of vector databases"

llm = ChatOpenAI(model_name="gpt-3.5-turbo")

def metadata_func(record: dict, metadata: dict) -> dict:
                metadata["name"] = record.get("name")
                metadata["id"] = record.get("worker_id", record.get("manager_id"))
                return metadata

data = []
loaders = []
current_dir = os.getcwd()
for file in os.listdir(os.path.join(current_dir, 'jsons')):
    if file.endswith('json'):
        file_path = os.path.join(current_dir, 'jsons', file)

        loader = JSONLoader(
                file_path=file_path,
                jq_schema = '.',
                text_content=False,
                metadata_func=metadata_func
            )
        docs = loader.load()
        for doc in docs:
            doc.metadata['file'] = file
        data.extend(docs)

if es.indices.exists(index="elastic_search_vectorstore"):
    es.indices.delete(index="elastic_search_vectorstore")
vectorstore = ElasticsearchStore.from_documents(
    documents=data,
    index_name="elastic_search_vectorstore",
    embedding=OpenAIEmbeddings(),
    es_url="http://localhost:9200",
)
#vectorstore = Chroma.from_documents(documents=text_docs+json_data, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})


human = '''TOOLS
------
Assistant can ask the user to use tools to look up information that may be helpful in answering the users original question. The tools the human can use are:

{tools}

RESPONSE FORMAT INSTRUCTIONS
----------------------------

When responding to me, please output a response in one of two formats:

**Option 1:**
Use this if you want the human to use a tool.
Markdown code snippet formatted in the following schema:

```json
{{
    "action": string, \ The action to take. Must be one of {tool_names}
    "action_input": string \ The input to the action
}}
```

**Option #2:**
Use this if you want to respond directly to the human. Markdown code snippet formatted in the following schema:

```json
{{
    "action": "Final Answer",
    "action_input": string \ You should put what you want to return to use here
}}
```

USER'S INPUT
--------------------
Here is the user's input (remember to respond with a markdown code snippet of a json blob with a single action, and NOTHING else):

{input}'''
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a powerful assistant who provides answers to questions based on retrieved data from memory, tools
        <tools>
        {tools}
        <tools>
        and tool names
        <tool_names>
        {tool_names}
        <tool_names> """),
        ("human", human),
        MessagesPlaceholder("chat_history", optional=True),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)
tool = create_retriever_tool(
    retriever = retriever,
    name = "json_retriever",
    description = "Searches and returns relevant data from a variety of input jsons to answer questions",
)

tools = [tool]
agent = create_json_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

#while (user_input := input("Enter a question (q to Quit): ")) != "q":
#    result = agent_executor.invoke({"input": user_input})
#    print(f'Question: {result["input"]}')
#    print(f'Answer: {result["output"]}')

"""
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(data)
if es.indices.exists(index="elastic_search_vectorstore"):
            es.indices.delete(index="elastic_search_vectorstore")
        vectorstore = ElasticsearchStore.from_documents(
            documents=text_docs+json_data,
            index_name="elastic_search_vectorstore",
            es_api_key="Mi00S1Y1QUJtaG1ydHk0OWNnWDU6OVZSVEVqQ1pRcnVyVzlpOXY0SExVdw==",
            embedding=OpenAIEmbeddings(),
            es_cloud_id="e78e4d9167bc43e7842f1f756114af3a:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvJGQxZTJiZmFhZTg3NjRiNzBhNDNhMjQ3NWJkNzRiYWMwJGNhZmM1YjdiNzIyNzRhM2NhYzdiYjJiYWM3MThiNjgx"
        )
#vectorstore = Chroma.from_documents(documents=docs, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
chunks = retriever.invoke(question)
print("---Retrival---:")
print(len(chunks))
for i in range(len(chunks)):
    chunk = chunks[i].page_content.replace("\n", " ")
    print(f'{i+1}: \n {chunk}\n')
print('---End Retrieval---')


def create_rag_chain(prompt=None):
    if prompt == None: 
        prompt = hub.pull("rlm/rag-prompt")
    else:
        prompt = PromptTemplate.from_template(prompt)
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt 
        | llm
        | StrOutputParser()
    )
    return rag_chain

print("---RAG with prompt from hub---:")
print(create_rag_chain().invoke(question))
print("---END RAG with prompt from hub---:")

print("---RAG with custom prompt---:")
#print(hub.pull("rlm/rag-prompt"))
custom_prompt = 
Using {context} as a retriever for data chucks relevant to {question}, polish the context to provide the answer to the question.
However if {question} is not strongly related to any of questions in {context}, say 
'I'm afraid I cannot assist with that. If you have any questions concering retrieval augmented generation, I would be happy to help'

print(create_rag_chain(custom_prompt).invoke(question))
print("---END RAG with custom prompt---:")


print("---Semantic Router Guard---")
rag_route = Route(
    name="Retrieval_Augmented_Generation",
    utterances=[
        "Overview of retrieval augmented generation",
        "How does retrieval augmented generation work",
        "What are retrieval augmented generation frameworks",
        "What is a text splitter",
        "What are vector databases",
        "How to implement retrieval augmented generation",
        "What are pros and cons of retrieval augmented generation",
        "How do you enhance retrieval augmented generation",
        "What are some examples of cloud computing services"
    ],
)

encoder = OpenAIEncoder()
routes = [rag_route]
layer = RouteLayer(encoder=encoder, routes=routes)

def semantic_layer(question):
    route = layer(question)

    if route.name == "Retrieval_Augmented_Generation":
        print(question + f" \nANSWER:")
        print(create_rag_chain().invoke(question))
    else:
        output = "I'm afraid I cannot assist with that. If you have any questions concering retrieval augmented generation, I would be happy to help"
        print(output)
        
semantic_layer(question)
print("---End Semantic Router Guard---")

prompt = hub.pull("rlm/rag-prompt")
print(prompt)
"""
