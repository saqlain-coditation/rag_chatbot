import os
from dotenv import load_dotenv
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.core import Settings

from rag.rag_debugger import RagDebugger

load_dotenv()

debug_handler = RagDebugger()
callback_manager = CallbackManager([debug_handler])
embedding = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
llm = GoogleGenAI(model="gemini-2.5-flash")
evaluator = GoogleGenAI(
    model="gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY_EVAL")
)

Settings.embed_model = embedding
Settings.callback_manager = callback_manager
