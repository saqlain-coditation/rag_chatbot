import os

from dotenv import load_dotenv
from google import genai
from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI

from rag.rag_debugger import RagDebugger

load_dotenv()

debug_handler = RagDebugger()
callback_manager = CallbackManager([debug_handler])
Settings.callback_manager = callback_manager

embedding = GoogleGenAIEmbedding(model_name="gemini-embedding-001")
llm = GoogleGenAI(model="gemini-2.0-flash")

vision_llm_model = "gemini-2.5-flash"
genai_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
