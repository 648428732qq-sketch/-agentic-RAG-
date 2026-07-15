import os
import uuid
import config
from db.vector_db_manager import VectorDbManager
from db.parent_store_manager import ParentStoreManager
from document_chunker import DocumentChuncker
from rag_agent.tools import ToolFactory
from rag_agent.graph import create_agent_graph
from core.observability import Observability


def _create_llm():
    """根据 ACTIVE_LLM_CONFIG 创建对应的 LLM 实例"""
    active = config.ACTIVE_LLM_CONFIG
    cfg = config.LLM_CONFIGS.get(active)
    if not cfg:
        raise ValueError(f"不支持的 LLM 提供商: {active}")

    model = cfg["model"]
    temperature = cfg["temperature"]
    model_kwargs = cfg.get("model_kwargs", {})

    if active == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, temperature=temperature)
    elif active == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, temperature=temperature)
    elif active == "deepseek":
        from langchain_openai import ChatOpenAI
        class DeepSeekChatOpenAI(ChatOpenAI):
            def bind_tools(self, tools, **kwargs):
                return super().bind_tools(tools, strict=False, **kwargs)
        return DeepSeekChatOpenAI(
            model=model, temperature=temperature,
            base_url="https://api.deepseek.com",
            api_key=os.environ.get("DEEPSEEK_API_KEY", "")
        )
    elif active == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, temperature=temperature)
    elif active == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, temperature=temperature)
    else:
        raise ValueError(f"不支持的 LLM 提供商: {active}")


class RAGSystem:

    def __init__(self, collection_name=config.CHILD_COLLECTION):
        self.collection_name = collection_name
        self.vector_db = VectorDbManager()
        self.parent_store = ParentStoreManager()
        self.chunker = DocumentChuncker()
        self.observability = Observability()
        self.agent_graph = None
        self.thread_id = str(uuid.uuid4())
        self.recursion_limit = config.GRAPH_RECURSION_LIMIT

    def initialize(self):
        self.vector_db.create_collection(self.collection_name)
        collection = self.vector_db.get_collection(self.collection_name)

        llm = _create_llm()
        print(f"  LLM 提供商: {config.ACTIVE_LLM_CONFIG} | 模型: {llm.model_name}")
        tools = ToolFactory(collection).create_tools()
        self.agent_graph = create_agent_graph(llm, tools)

    def get_config(self):
        cfg = {"configurable": {"thread_id": self.thread_id}, "recursion_limit": self.recursion_limit}
        handler = self.observability.get_handler()
        if handler:
            cfg["callbacks"] = [handler]
        return cfg

    def reset_thread(self):
        try:
            self.agent_graph.checkpointer.delete_thread(self.thread_id)
        except Exception as e:
            print(f"Warning: Could not delete thread {self.thread_id}: {e}")
        self.thread_id = str(uuid.uuid4())