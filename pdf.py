import os
from llama_index.core import load_index_from_storage, StorageContext, VectorStoreIndex
from llama_index.readers.file import PDFReader
import Model1
#os.environ["OPENAI_API_KEY"] = ""


def get_index(data,index_name):
    index = None
    if not os.path.exists(index_name):
        print("Building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index

def get_engine(path, name):
    pdf = PDFReader().load_data(file=path)
    index = get_index(pdf, name)
    return index.as_query_engine()

