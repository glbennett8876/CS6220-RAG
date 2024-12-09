from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken

# Token Estimation Function
def estimate_tokens(text, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))

# Function to Split Large Documents
def split_large_document(document, max_tokens, encoding_name="cl100k_base"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_tokens, chunk_overlap=0)
    chunks = text_splitter.split_text(document.page_content)
    return [{"page_content": chunk, "metadata": document.metadata} for chunk in chunks]

# Truncate Documents with Splitting
def truncate_documents(documents, question, max_tokens=2048, single_doc_limit=500):
    encoding_name = "cl100k_base"
    question_tokens = estimate_tokens(question, encoding_name)
    remaining_tokens = max_tokens - question_tokens

    truncated_docs = []
    current_tokens = 0
    for doc in documents:
        doc_tokens = estimate_tokens(doc.page_content, encoding_name)

        # If a single document exceeds the limit, split it into smaller chunks
        if doc_tokens > single_doc_limit:
            split_docs = split_large_document(doc, single_doc_limit, encoding_name)
            for split_doc in split_docs:
                split_doc_tokens = estimate_tokens(split_doc["page_content"], encoding_name)
                if current_tokens + split_doc_tokens <= remaining_tokens:
                    truncated_docs.append(split_doc)
                    current_tokens += split_doc_tokens
                else:
                    break
        else:
            # Add the whole document if it fits
            if current_tokens + doc_tokens <= remaining_tokens:
                truncated_docs.append(doc)
                current_tokens += doc_tokens
            else:
                break
    return truncated_docs

class TruncatingRetriever:
    def __init__(self, retriever, max_tokens=2048, single_doc_limit=500):
        print("here i am building")
        self.retriever = retriever
        self.max_tokens = max_tokens
        self.single_doc_limit = single_doc_limit

    def get_relevant_documents(self, query):
        print("GETTING DOCS")
        # Retrieve documents
        documents = self.retriever.get_relevant_documents(query)
        # Truncate based on token limit with splitting
        return truncate_documents(documents, query, max_tokens=self.max_tokens, single_doc_limit=self.single_doc_limit)
