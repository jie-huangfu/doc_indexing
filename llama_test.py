from llama_index.core import SimpleDirectoryReader

docs = SimpleDirectoryReader(input_dir="./data").load_data()


from llama_index.core import Document
from llama_index.core.schema import MetadataMode

document = Document(
    text="This is a super-customized document",
    metadata={
        "file_name": "super_secret_document.txt",
        "category": "finance",
        "author": "LlamaIndex",
    },
    # excluded_embed_metadata_keys=["file_name"],
    excluded_llm_metadata_keys=["category"],
    metadata_seperator="\n",
    metadata_template="{key}:{value}",
    text_template="Metadata:\n{metadata_str}\n-----\nContent:\n{content}",
)

print(
    "The LLM sees this: \n",
    document.get_content(metadata_mode=MetadataMode.LLM),
)

from llama_index.core.schema import MetadataMode

# print(docs[0].get_content(metadata_mode=MetadataMode.LLM))   # what the llm sees
print(docs[0].get_content(metadata_mode=MetadataMode.EMBED)) # what embeddings see. in this case, same thing

for doc in docs:
    # define the content/metadata template
    doc.text_template = "Metadata:\n{metadata_str}\n---\nContent:\n{content}"

    # exclude page label from embedding
    if "page_label" not in doc.excluded_embed_metadata_keys:
        doc.excluded_embed_metadata_keys.append("page_label")

print(docs[0].get_content(metadata_mode=MetadataMode.EMBED))

from llama_index.llms.groq import Groq
import os
import getpass

os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

llm_transformations = Groq(model="qwen-2.5-32b", api_key=os.environ["GROQ_API_KEY"])

from llama_index.core.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.core.node_parser import SentenceSplitter

text_splitter = SentenceSplitter(
    separator=" ", chunk_size=1024, chunk_overlap=128
)
title_extractor = TitleExtractor(llm=llm_transformations, nodes=5)
qa_extractor = QuestionsAnsweredExtractor(llm=llm_transformations, questions=3)


from llama_index.core.ingestion import IngestionPipeline

pipeline = IngestionPipeline(
    transformations=[
        text_splitter,
        title_extractor,
        qa_extractor
    ]
)

nodes = pipeline.run(
    documents=docs,
    in_place=True,
    show_progress=True,
)

len(nodes)


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

hf_embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

test_embed = hf_embeddings.get_text_embedding("Hello world")
print(test_embed)

from llama_index.core import VectorStoreIndex

index = VectorStoreIndex(nodes, embed_model=hf_embeddings)

"""## Query"""

llm_querying = Groq(model="llama-3.3-70b-versatile", api_key=os.environ["GROQ_API_KEY"])

query_engine = index.as_query_engine(llm=llm_querying)
response = query_engine.query(
    "what does this model do?"
)


index.storage_context.persist(persist_dir="./vectors")

from llama_index.core import StorageContext, load_index_from_storage

# rebuild storage context
storage_context = StorageContext.from_defaults(persist_dir="./vectors")

# load index
index_from_storage = load_index_from_storage(storage_context, embed_model=hf_embeddings)

qa = index_from_storage.as_query_engine(llm=llm_querying)

response = qa.query("what does this model do?")
print(response)

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

db = chromadb.PersistentClient(path="./chroma_db")

chroma_collection = db.get_or_create_collection("healthGPT")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex(
    nodes, storage_context=storage_context, embed_model=hf_embeddings
)

query_engine = index.as_query_engine(llm=llm_querying)

response = query_engine.query("What is this model good at?")
print(response)