import os
import sys
import shutil
import chromadb
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# 将 backend 路径添加到 sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend', 'app'))
from core.config import settings

KNOWLEDGE_BASE_DIR = "knowledge_base"
COLLECTION_NAME = "interview_assistant"

def clean_chroma_data(chroma_data_path):
    """
    完全清理ChromaDB数据目录
    """
    if os.path.exists(chroma_data_path):
        print(f"🗑️ 清理现有 ChromaDB 数据目录: {chroma_data_path}")
        try:
            shutil.rmtree(chroma_data_path)
            print("✅ 成功清理旧数据")
        except Exception as e:
            print(f"⚠️ 警告: 无法清理旧数据: {e}")
    
    # 重新创建目录
    os.makedirs(chroma_data_path, exist_ok=True)

def main():
    print("🚀 开始知识库摄取...")

    # 定义 ChromaDB 数据存储路径，与 rag_service.py 中的路径一致
    chroma_data_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), "chroma_data")
    
    # 完全清理数据目录（推荐用于开发阶段）
    clean_chroma_data(chroma_data_path)
    
    print(f"摄取脚本使用 ChromaDB (持久化客户端) 路径: {chroma_data_path}")
    # 初始化 ChromaDB 的持久化客户端
    client = chromadb.PersistentClient(path=chroma_data_path)

    # 1. 加载文档
    all_documents = []

    try:
        pdf_loader = DirectoryLoader(
            KNOWLEDGE_BASE_DIR,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        all_documents.extend(pdf_loader.load())
        print(f"📕 加载了 {len(all_documents)} PDF 文档") # 修正计数
    except Exception as e:
        print(f"⚠️ 加载 PDF 文件时出错: {e}")

    try:
        text_loader = DirectoryLoader(
            KNOWLEDGE_BASE_DIR,
            glob="**/*.txt",
            loader_cls=TextLoader,
            show_progress=True,
            use_multithreading=True
        )
        txt_docs = text_loader.load()
        all_documents.extend(txt_docs)
        print(f"📄 加载了 {len(txt_docs)} TXT 文档")
    except Exception as e:
        print(f"⚠️ 加载 TXT 文件时出错: {e}")

    try:
        docx_loader = DirectoryLoader(
            KNOWLEDGE_BASE_DIR,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True,
            use_multithreading=True
        )
        docx_docs = docx_loader.load()
        all_documents.extend(docx_docs)
        print(f"📘 加载了 {len(docx_docs)} DOCX 文档")
    except Exception as e:
        print(f"⚠️ 加载 DOCX 文件时出错: {e}")

    documents = all_documents

    if not documents:
        print(f"❌ 在 {KNOWLEDGE_BASE_DIR} 目录中未找到文档。中止。")
        return

    print(f"📚 总共加载的文档数量: {len(documents)}")

    # 2. 将文档分割成块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"📄 分割成 {len(chunks)} 个块。")

    # 3. 初始化 LangChain 的 HuggingFace Embeddings
    print("🧠 使用 LangChain HuggingFaceEmbeddings 创建嵌入...")
    embeddings_for_langchain_chroma = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # 4. 存储到 ChromaDB (使用持久化客户端)
    print(f"💾 将块存储到 ChromaDB 集合: {COLLECTION_NAME}...")

    try:
        # 使用 LangChain 的 Chroma.from_documents 来创建并填充向量存储
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings_for_langchain_chroma,
            collection_name=COLLECTION_NAME,
            client=client,
            persist_directory=chroma_data_path
        )

        # 获取集合的实际文档数量进行确认
        document_count = vector_store._collection.count()
        print(f"✅ 摄取完成! 集合中的总文档数量: {document_count}")
        
        # 验证数据完整性
        if document_count != len(chunks):
            print(f"⚠️ 警告: 预期 {len(chunks)} 个文档但找到 {document_count} 个")
        
    except Exception as e:
        print(f"❌ 摄取过程中出错: {e}")
        raise

if __name__ == "__main__":
    main()
