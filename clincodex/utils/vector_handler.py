import os
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from clincodex.utils.base_handler import load_sentence_transformer

def create_vector_store(data_list,vector_key,spec_map):
    if not os.path.isdir(spec_map["VECTOR_DB_PATHS"][vector_key]):
        metadata_list = list(map(lambda x:{"source": x + " from ICD knowledge graph"}, data_list))

        start_time = time.time()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=spec_map["CHUNK_SIZE"], chunk_overlap=spec_map["CHUNK_OVERLAP"])
        docs = text_splitter.create_documents(data_list, metadatas=metadata_list)
        batches = [docs[i:i + spec_map["BATCH_SIZE"]] for i in range(0, len(docs), spec_map["BATCH_SIZE"])]

        vectorstore = Chroma(embedding_function=load_sentence_transformer(model_name=spec_map["SENTENCE_EMBEDDING_MODEL"]), 
                                persist_directory=spec_map["VECTOR_DB_PATHS"][vector_key])
                                # collection_metadata={"hnsw:space": "cosine"})

        for batch in batches:
            vectorstore.add_documents(documents=batch)

        end_time = round((time.time() - start_time)/(60), 2)

        print("VectorDB is created in {} mins".format(end_time))
    else:
        print("VectorDB is already exists @ {0} ".format(spec_map["VECTOR_DB_PATHS"][vector_key]))

def load_vector_store(vector_key, spec_map):
    embedding_function = load_sentence_transformer(spec_map["SENTENCE_EMBEDDING_MODEL_FOR_NODE_RETRIEVAL"])
    return Chroma(persist_directory=spec_map["VECTOR_DB_PATHS"][vector_key], embedding_function=embedding_function)
        