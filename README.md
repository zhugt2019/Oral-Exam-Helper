# Oral-Exam-Helper

使用方法：

1. git clone https://github.com/zhugt2019/Oral-Exam-Helper.git
2. 将Gemini API设为系统变量，或者使用Ollama
3. cd Oral-Exam-Helper
4. pip install -r requirements.txt
5. 将文档放入Oral-Exam-Helper\knowledge_base文件夹
6. 在第一个终端窗口中（记得先保证路径仍然为Oral-Exam-Helper）：
7. python scripts\ingest.py
8. 在第二个终端窗口中（记得先cd Oral-Exam-Helper）：
10. uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
11. 后端！启动！
12. streamlit run frontend\app.py
13. 前端！启动！
