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

# 确保集合名与 ingest.py 中一致
COLLECTION_NAME = "interview_assistant"

class RAGService:
    def __init__(self):
        # 定义 ChromaDB 数据存储的路径，数据将存储在项目根目录下的 'chroma_data' 文件夹中
        chroma_data_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')), "chroma_data")
        os.makedirs(chroma_data_path, exist_ok=True) # 确保目录存在

        print(f"RAGService connecting to ChromaDB (Persistent Client) at: {chroma_data_path}")
        # 使用 PersistentClient 来创建一个持久化的 ChromaDB 实例
        self.client = chromadb.PersistentClient(path=chroma_data_path)

        # 使用 HuggingFace Embeddings
        self.embedding_function = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("[RAGService] LangChain's HuggingFaceEmbeddings initialized for RAGService.")

        # 初始化向量存储器
        self.vector_store = Chroma(
            client=self.client,
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_function
        )

        # 尝试获取集合的实际文档数量进行确认
        try:
            actual_collection = self.client.get_collection(name=COLLECTION_NAME)
            print(f"ChromaDB collection '{COLLECTION_NAME}' document count: {actual_collection.count()}")
        except Exception as e:
            print(f"Could not get collection '{COLLECTION_NAME}' count: {e}")
            print("Please run ingest.py first to populate the knowledge base.")

        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})

        # 定义 prompt 模板
        template = """
        You are an AI Interview Assistant. Use the following retrieved context to answer the user's question.
        Provide a concise and professional answer based on the provided context.
        After the answer, list the sources you used with their metadata.

        CONTEXT:
        {context}

        QUESTION:
        {question}

        ANSWER:
        """
        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    def _format_docs(self, docs):
        """格式化文档并创建来源列表。"""
        formatted_context = "\n\n".join(
            f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}"
            for doc in docs
        )

        unique_sources = set()
        for doc in docs:
            source_path = doc.metadata.get('source', 'Unknown')
            page_number = doc.metadata.get('page', 'N/A')
            unique_sources.add((source_path, page_number))

        sources = "\n".join(
            f"- {os.path.basename(src_path)}, Page: {pg_num}"
            for src_path, pg_num in sorted(list(unique_sources))
        )
        return formatted_context, sources

    def get_rag_chain(self, model_provider: str):
        llm = get_llm(model_provider)

        rag_chain_core = (
            self.prompt
            | llm
            | StrOutputParser()
        )
        return rag_chain_core

    async def invoke_chain(self, question: str, model_provider: str):
        docs = self.retriever.invoke(question)
        
        if not docs:
            return {"answer": "I could not find any relevant information in the knowledge base to answer your question.", "sources": "No sources found."}

        formatted_context, sources_text = self._format_docs(docs)

        chain = self.get_rag_chain(model_provider)

        result = chain.invoke({
            "question": question,
            "context": formatted_context,
        })

        return {"answer": result, "sources": sources_text}

rag_service = RAGService()
