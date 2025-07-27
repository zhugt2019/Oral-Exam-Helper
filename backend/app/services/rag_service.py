# backend/app/services/rag_service.py
import os
import chromadb
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

# 导入 LangChain 的 HuggingFace Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
# 导入 LangChain 的 Chroma 向量存储
from langchain_community.vectorstores import Chroma

# 导入你的 LLM 服务和配置
from .llm_service import get_llm
from ..core.config import settings

# 导入日志模块
import logging
logger = logging.getLogger(__name__)
import torch # 导入torch以检查CUDA可用性

# 确保集合名与 ingest.py 中一致
COLLECTION_NAME = "interview_assistant"

class RAGService:
    def __init__(self):
        # 定义 ChromaDB 数据存储的路径，数据将存储在项目根目录下的 'chroma_data' 文件夹中
        chroma_data_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')), "chroma_data")
        os.makedirs(chroma_data_path, exist_ok=True) # 确保目录存在

        logger.info(f"RAGService connecting to ChromaDB (Persistent Client) at: {chroma_data_path}")
        # 使用 PersistentClient 来创建一个持久化的 ChromaDB 实例
        self.client = chromadb.PersistentClient(path=chroma_data_path)

        # 检查集合是否存在，如果不存在，创建一个 (这由get_or_create_collection处理)
        try:
            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
            logger.info(f"ChromaDB collection '{COLLECTION_NAME}' document count: {self.collection.count()}")
            if self.collection.count() == 0:
                logger.warning(f"ChromaDB collection '{COLLECTION_NAME}' is empty. Please run ingest.py to add documents.")
        except Exception as e:
            logger.error(f"Error connecting to ChromaDB collection '{COLLECTION_NAME}': {e}", exc_info=True)
            self.collection = None # 如果连接失败，将 collection 设为 None，以防后续操作报错

        # 动态选择 HuggingFace Embeddings 的设备 (CPU 或 CUDA)
        embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing HuggingFaceEmbeddings on device: {embedding_device}")
        self.embedding_function = HuggingFaceEmbeddings( # 属性名修正为 embedding_function
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': embedding_device}, # 动态选择设备
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("[RAGService] LangChain's HuggingFaceEmbeddings initialized for RAGService.")


        # 初始化 ChromaDB 作为向量存储
        self.vector_store = Chroma(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_function, # 确保这里使用正确的属性名
            # persist_directory 必须与 client 的 path 一致，确保正确持久化
            persist_directory=chroma_data_path
        )

        # 初始化检索器
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 3}) # 检索前3个最相关的文档

        # 定义 RAG 提示模板
        self.prompt = PromptTemplate.from_template("""
        根据以下上下文信息，简洁、准确地回答问题。
        如果上下文无法提供答案，请说明“我无法从提供的知识库中找到答案”。
        请避免臆造信息。

        上下文:
        {context}

        问题: {question}
        回答:
        """)

    def _format_docs(self, docs):
        """格式化检索到的文档，用于构建上下文和来源信息。"""
        formatted_context = "\\n\\n".join(
            f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\\nContent: {doc.page_content}"
            for doc in docs
        )

        unique_sources = set()
        for doc in docs:
            source_path = doc.metadata.get('source', 'Unknown')
            page_number = doc.metadata.get('page', 'N/A')
            unique_sources.add((source_path, page_number))

        sources = "\\n".join(
            f"- {os.path.basename(src_path)}, Page: {pg_num}"
            for src_path, pg_num in sorted(list(unique_sources))
        )
        return formatted_context, sources

    def get_rag_chain(self, model_provider: str):
        """根据模型提供者获取 RAG 链。"""
        llm = get_llm(model_provider)

        rag_chain_core = (
            self.prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain_core

    async def invoke_chain(self, question: str, model_provider: str):
        """异步调用 RAG 链进行问答。"""
        if self.collection and self.collection.count() == 0:
            logger.warning("ChromaDB collection is empty, returning default 'no context' answer.")
            return {"answer": "I could not find any relevant information in the knowledge base to answer your question. The knowledge base is currently empty.", "sources": "No sources found."}

        docs = self.retriever.invoke(question)

        if not docs:
            logger.warning(f"No relevant documents found for question: '{question}'. Returning default answer.")
            return {"answer": "I could not find any relevant information in the knowledge base to answer your question.", "sources": "No sources found."}

        formatted_context, sources_text = self._format_docs(docs)

        chain = self.get_rag_chain(model_provider)

        logger.info(f"[RAGService] Invoking chain for question: '{question}' with context length: {len(formatted_context)}...")
        try:
            # 使用 await chain.ainvoke() 因为 rag_service.invoke_chain 是异步的
            raw_llm_result = await chain.ainvoke({
                "question": question,
                "context": formatted_context,
            })
            logger.info(f"[RAGService] Raw LLM chain result type: {type(raw_llm_result)}, value (first 200 chars): {str(raw_llm_result)[:200]}")

            # 确保结果是字符串，并处理空字符串情况
            if not isinstance(raw_llm_result, str):
                logger.error(f"LLM chain returned non-string type: {type(raw_llm_result)}, value: {raw_llm_result}. Returning default error message.")
                return {"answer": "Error: LLM did not return a valid string response (type mismatch).", "sources": "N/A"}
            elif not raw_llm_result.strip(): # 检查是否为空字符串或只包含空格
                logger.warning("LLM chain returned an empty or whitespace string. Returning default answer.")
                return {"answer": "I could not generate a meaningful answer based on the provided information.", "sources": "N/A"}

            return {"answer": raw_llm_result, "sources": sources_text}

        except Exception as e:
            logger.error(f"Error during LLM chain invocation: {e}", exc_info=True) # 打印完整的异常信息
            return {"answer": f"Error during answer generation: {str(e)}", "sources": "N/A"}

# 实例化服务
rag_service = RAGService()
