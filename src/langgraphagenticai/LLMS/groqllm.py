import os
from typing import Optional
from langchain_groq import ChatGroq

class GroqLLM:
    def __init__(self, model: str = "openai/gpt-oss-20b", api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set in environment.")
        self.model = model
        try:
            self._llm = ChatGroq(api_key=self.api_key, model=self.model)
        except Exception as e:
            raise ValueError(f"Failed to initialize ChatGroq: {e}")

    def get_llm_model(self):
        return self._llm