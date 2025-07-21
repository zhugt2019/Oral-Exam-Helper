# Oral-Exam-Helper

使用方法：

1. git clone https://github.com/zhugt2019/Oral-Exam-Helper.git
2. 将Gemini API设为系统变量，或者使用Ollama
3. 将文档放入Oral-Exam-Helper\knowledge_base文件夹
4. cd Oral-Exam-Helper
5. python scripts\ingest.py
6. uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
7. streamlit run frontend\app.py
8. 启动！
