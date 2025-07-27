# Oral-Exam-Helper

使用方法：

1. git clone https://github.com/zhugt2019/Oral-Exam-Helper.git
2. 将Gemini API设为系统变量，或者使用Ollama
3. （重要）启用立体声混音：根据操作系统不同，启用方法也可能不同，可自行查询
4. cd Oral-Exam-Helper
5. pip install -r requirements.txt
6. 将文档放入Oral-Exam-Helper\knowledge_base文件夹
7. 在第一个终端窗口中（记得先保证路径仍然为Oral-Exam-Helper）：
8. python scripts\ingest.py
9. 在第二个终端窗口中（记得先cd Oral-Exam-Helper）：
10. uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
11. 后端！启动！
12. python -m desktop_app.main_gui
13. （暂时弃置）streamlit run frontend\app.py
14. 前端！启动！

备注：

desktop_app\stt_processor.py中可调节参数：
- vad_filter: True或False，决定是否过滤静音和背景噪音
- model_size：可根据硬件和需求尝试tiny -> base -> small -> medium -> large
- language：默认en

待办：

- STT processor stopped （结束识别）之后无法重新开始新的识别
- 识别过程中页面自动刷新会导致文字难以选取
- 前端整体有待优化
