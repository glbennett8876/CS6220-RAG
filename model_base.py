import os
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_elasticsearch import ElasticsearchStore
from langchain.docstore.document import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.chat_message_histories import ChatMessageHistory
from elasticsearch import Elasticsearch
from semantic_router import Route
from pydantic import Field


class DPEmbeddings(OpenAIEmbeddings):
    epsilon: float = Field(1.0, description="Differential privacy budget for embedding noise")  # Declare epsilon as a class field

    def embed_text(self, text):
        original_embedding = super().embed_text(text)
        sensitivity = 1.0  # Assume unit sensitivity for embeddings
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, len(original_embedding))
        noisy_embedding = original_embedding + noise
        return noisy_embedding


class DPChatOpenAI(ChatOpenAI):
    epsilon: float = Field(0.5, description="Differential privacy budget for logit noise")  # Declare epsilon as a class field

    def _add_noise_to_logits(self, logits):
        sensitivity = 1.0  # Assume unit sensitivity for logits
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, logits.shape)
        return logits + noise

    def generate_response(self, prompt):
        response = super().generate(prompt)
        response_logits = self._add_noise_to_logits(response["logits"])
        response["logits"] = response_logits
        return response


class Model:
    def __init__(self):
        os.environ['OPENAI_API_KEY'] = ""
        self.question = None
        self.llm = DPChatOpenAI(model_name="gpt-4", epsilon=0.5)  # No changes needed here
        self.retriever = None
        self.es = Elasticsearch('http://localhost:9200')
        self.routes = []
        self.memory = ChatMessageHistory()

    def add_route(self, name, utterances):
        route = Route(name=name, utterances=utterances)
        self.routes.append(route)

    def get_retriever(self, json_data=[]):
        # Reindex with DP embeddings
        if self.es.indices.exists(index="elastic_search_vectorstore"):
            self.es.indices.delete(index="elastic_search_vectorstore")
        # embedding = EmbeddingModelA()
        # embedding = EmbeddingModelB()
        # embedding = EmbeddingModelC()
        # embedding = OpenAIEmbeddings()
        embedding = DPEmbeddings(epsilon=0.5)
        vectorstore = ElasticsearchStore.from_documents(
            documents=json_data,
            index_name="elastic_search_vectorstore",
            embedding=embedding,  # Pass epsilon explicitly
            es_url="http://localhost:9200",
        )
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    def create_rag_chain(self):
        prompt_template = """
        You are a powerful assistant who provides answers to questions based on retrieved data using context:
        <context>
        {context}
        </context>
        Question: {question}
        Make sure to describe your chain of thought process for your answers.
        """
        prompt = PromptTemplate.from_template(prompt_template)

        rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(self.question)

    def router(self):
        prompt = PromptTemplate(
            input_variables=["question"],
            template="""Classify the following question into one of the following: Self, Team
            Use the following to define the classification:
            'Self': The user is asking for information relevant to their own data.
            'Team': The user is asking for information relevant to one or more of their employees or their entire team.
            Question: {question}
            Answer in one word.
            """
        )
        route_chain = {"question": RunnablePassthrough()} | prompt | self.llm
        route = route_chain.invoke(self.question).content
        return route
    from sentence_transformers import SentenceTransformer
class EmbeddingModelA:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """
        Embeds a list of documents.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, query):
        """
        Embeds a single query.
        """
        return self.model.encode([query], convert_to_numpy=True).tolist()[0]
    
class EmbeddingModelB:
    def __init__(self, model_name="paraphrase-mpnet-base-v2"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """
        Embeds a list of documents.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, query):
        """
        Embeds a single query.
        """
        return self.model.encode([query], convert_to_numpy=True).tolist()[0]
    
class EmbeddingModelC:
    def __init__(self, model_name="all-distilroberta-v1"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        """
        Embeds a list of documents.
        """
        if isinstance(texts, str):
            texts = [texts]
        return self.model.encode(texts, convert_to_numpy=True).tolist()

    def embed_query(self, query):
        """
        Embeds a single query.
        """
        return self.model.encode([query], convert_to_numpy=True).tolist()[0]    