# AYesha — DDS Enterprise HR Chatbot (Hugging Face Space, ZeroGPU)
# !pip install -U gradio spaces pinecone llama-index llama-index-vector-stores-pinecone llama-index-readers-file pypdf llama-index-llms-openai llama-index-embeddings-openai

# --- Imports ---
import logging
import os
import sys

import gradio as gr
import spaces

from pinecone import Pinecone, ServerlessSpec
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
    PromptTemplate,
)
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.readers.file import PDFReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# --- Logging ---
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Keys (set these in Space Settings -> Variables and secrets) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Add it under Settings -> Variables and secrets.")
if not PINECONE_API_KEY:
    raise RuntimeError("PINECONE_API_KEY is not set. Add it under Settings -> Variables and secrets.")

# --- LlamaIndex global settings ---
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.2)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
Settings.chunk_size = 600
Settings.chunk_overlap = 200

SYSTEM_PROMPT = """You are AYesha, the Decoding Data Science (DDS) Enterprise HR Chatbot. Answer questions exclusively using the attached DDS HR Handbook. Base all responses on the most up-to-date information available in the handbook. Only respond to queries directly related to DDS HR policies as outlined in the handbook.

Important instructions:
- Only answer questions directly supported by the latest DDS HR handbook.
- Decline politely and redirect to the provided email address for any questions outside scope or for confidential information.
- Always reason before concluding. Only present the answer after checking scope and source.
Remember: As AYesha, the DDS HR Enterprise Chatbot, you must never provide information outside authorized HR handbook content and always respond respectfully according to these constraints.
"""

# LlamaIndex's query engine doesn't accept a `system_prompt=` kwarg — that param only
# exists on as_chat_engine(). The equivalent for a query engine is a custom QA prompt
# template, which is what actually enforces AYesha's scope/persona.
QA_TEMPLATE = PromptTemplate(
    SYSTEM_PROMPT
    + "\n\nContext information is below.\n---------------------\n{context_str}\n---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\nAnswer: "
)

# --- Initialize Pinecone ---
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "quickstart"
DIMENSION = 1536

existing_indexes = [idx["name"] for idx in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    # Only create the index the first time. Deleting + recreating on every
    # restart re-embeds every PDF (real API cost) and risks a race if two
    # cold starts overlap — so we only build the index when it doesn't exist yet.
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="euclidean",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(INDEX_NAME)
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

# If the index is already populated (from a prior run), just attach to it.
# Otherwise load the PDFs and build it now.
stats = pinecone_index.describe_index_stats()
if stats.get("total_vector_count", 0) > 0:
    index = VectorStoreIndex.from_vector_store(vector_store)
else:
    documents = SimpleDirectoryReader(
        input_dir="Data",  # folder name is case-sensitive — must match exactly what you upload
        required_exts=[".pdf"],
        file_extractor={".pdf": PDFReader()},
    ).load_data()

    if not documents:
        raise ValueError(
            "No PDF documents were loaded from the 'Data' folder. "
            "Make sure a folder named exactly 'Data' with your PDF(s) is uploaded alongside app.py."
        )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

query_engine = index.as_query_engine(text_qa_template=QA_TEMPLATE)


# --- Gradio App ---
@spaces.GPU
def query_doc(prompt):
    try:
        response = query_engine.query(prompt)
        return str(response)
    except Exception as e:
        logger.exception("Query failed")
        return f"Error: {str(e)}"


gr.Interface(
    fn=query_doc,
    inputs=gr.Textbox(label="Ask a question about the document"),
    outputs=gr.Textbox(label="Answer"),
    title="DDS Enterprise HR Chatbot — AYesha",
    description="Ask questions related to HR for latest information.",
).launch()
