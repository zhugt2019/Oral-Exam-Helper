import streamlit as st
import requests
from streamlit_mic_recorder import mic_recorder
import time
import re

st.set_page_config(page_title="AI Interview Assistant", layout="wide")

st.title("🤖 AI 面试助手")
st.markdown("上传您的知识库文档 (PDF, TXT, DOCX)，然后开始语音或文字面试。")

BACKEND_URL = "http://localhost:8000/api/v1"

# State Management
if "messages" not in st.session_state:
    st.session_state.messages = []
if "backend_status" not in st.session_state:
    st.session_state.backend_status = "checking"

# Backend Status Check Function
def check_backend_status():
    """Checks if the backend is reachable."""
    try:
        response = requests.get("http://localhost:8000/status", timeout=5)
        if response.status_code == 200:
            st.session_state.backend_status = "connected"
        else:
            st.session_state.backend_status = "disconnected"
    except requests.exceptions.ConnectionError:
        st.session_state.backend_status = "disconnected"
    except requests.exceptions.Timeout:
        st.session_state.backend_status = "timeout"
    except Exception as e:
        st.session_state.backend_status = f"error: {e}"

# Run backend status check once on app load
if st.session_state.backend_status == "checking":
    check_backend_status()
    if st.session_state.backend_status != "checking":
        st.rerun()

# UI Components
col1, col2 = st.columns(2)

with col1:
    st.header("交互控制")
    if st.session_state.backend_status == "connected":
        st.success("✅ 后端已连接")
    elif st.session_state.backend_status == "disconnected":
        st.error("❌ 后端未连接，请确保后端服务已启动。")
    elif st.session_state.backend_status == "timeout":
        st.warning("⚠️ 后端连接超时，可能运行缓慢或网络问题。")
    else:
        st.info(f"ℹ️ 检查后端时发生错误: {st.session_state.backend_status}")

    st.markdown("---")

    model_provider = st.selectbox(
        "选择语言模型 (LLM):",
        ("qwen", "gemini")
    )

    st.write("请通过语音提问:")
    audio_info = mic_recorder(
        start_prompt="⏺️ 开始录音",
        stop_prompt="⏹️ 停止录音",
        just_once=True,
        key='my_mic'
    )

    # 清空聊天记录按钮
    if st.button("清空聊天记录"):
        st.session_state.messages = []
        st.rerun()

with col2:
    st.header("聊天记录")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("查看引用来源"):
                    st.text(message["sources"])
            if "elapsed_time" in message and message["role"] == "assistant":
                st.caption(f"生成耗时: {message['elapsed_time']:.2f} 秒")

def handle_response(response_json, elapsed_time=None, role="assistant"):
    """Helper to process and display response."""
    answer = response_json.get("answer", "抱歉，发生错误。")
    sources = response_json.get("sources", "")

    # 移除 <think> 标签及其内容
    cleaned_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
    cleaned_answer = cleaned_answer.strip()

    message_data = {"role": role, "content": cleaned_answer, "sources": sources}
    if elapsed_time is not None:
        message_data["elapsed_time"] = elapsed_time

    st.session_state.messages.append(message_data)
    st.rerun()

# Logic for handling interactions

# Handle audio input
if audio_info:
    if st.session_state.backend_status not in ["connected", "timeout"]:
        st.error("后端未连接或存在问题，无法发送语音请求。")
        st.rerun()
    else:
        st.audio(audio_info['bytes'])
        audio_bytes = audio_info['bytes']

        st.session_state.messages.append({"role": "user", "content": "[语音已发送，后端正在处理...]"})

        files = {"audio_file": ("user_audio.wav", audio_bytes, "audio/wav")}
        data = {"model_provider": model_provider}

        try:
            start_time = time.time()
            with st.spinner("语音识别并生成答案中..."):
                response = requests.post(f"{BACKEND_URL}/chat/audio", files=files, data=data, timeout=120)
            elapsed_time = time.time() - start_time

            st.session_state.messages.pop()

            if response.status_code == 200:
                handle_response(response.json(), elapsed_time=elapsed_time)
            else:
                st.error(f"错误: {response.status_code} - {response.text}")
                st.session_state.messages.append({"role": "assistant", "content": "处理请求时发生错误。"})
                st.rerun()

        except requests.exceptions.RequestException as e:
            st.session_state.messages.pop()
            st.error(f"连接后端失败: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"请求后端时发生网络错误: {e}"})
            st.rerun()

# Handle text input
prompt = st.chat_input("或者在这里输入您的问题...")

if prompt:
    if st.session_state.backend_status not in ["connected", "timeout"]:
        st.error("后端未连接或存在问题，无法发送文本请求。")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})

        data = {"question": prompt, "model_provider": model_provider}
        try:
            start_time = time.time()
            with st.spinner("正在生成答案..."):
                response = requests.post(f"{BACKEND_URL}/chat/text", data=data, timeout=120)
            elapsed_time = time.time() - start_time

            if response.status_code == 200:
                handle_response(response.json(), elapsed_time=elapsed_time)
            else:
                st.error(f"错误: {response.status_code} - {response.text}")
                st.session_state.messages.append({"role": "assistant", "content": "处理请求时发生错误。"})
                st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"连接后端失败: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"请求后端时发生网络错误: {e}"})
            st.rerun()
