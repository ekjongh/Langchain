# ======================================================================================================================
# Streamlit + LangChain + ChatGPT + Chroma DB
# 출처: https://medium.com/@daydreamersjp/implementing-local-chatgpt-using-streamlit-3228bfab8ae7
# langchain structure (2023.09.08)
# > langchain
#   ├- chroma_db : 인베딩벡터를 저장하는 디렉토리
#   |     ├- index
#   ├- pdf : PDF 파일을 저장하는 디렉토리
#   ├- .env : 환경변수를 설정하는 파일
#   ├- chatgpt_app.py : 챗팅형식의 UI를 제공하는 모듈
#   ├- chatgpt_logger.py : 로그를 기록하는 모듈
#   ├- chatgpt_service.py : 챗팅(RAG) 서비스를 제공하는 모듈
#   ├- 문서들을 임베팅벡터로 변환_XXXX-X.ipynb : 작성된 모듈을 테스트하는 Jupyter Notebook
#   ├- crud : API 서비스를 위한 모듈 ( 비즈니스 모듈 - 대부분의 작업 대상임)
#   ...
# [ 실행 명령 ]
# > streamlit run chatgpt_app.py
#
# [ 문서들을 임베딩벡터로 변환하는 방법 ]
# > 주피터 노트북 사용하기 - 문서크기가 큰 경우 시간이 오래걸림
# ----------------------------------------------------------------------------------------------------------------------
# 2023.09.08 - 초기모듈 작성 (채팅형식 UI로 변경)
# 2023.09.09 - 가상환경 파이썬 버전 업그레이드(기존 3.9 -> 3.10)
#            - 사용자 챗GPT UI 변경 (기존소스 막짠 것 같아서)
# ======================================================================================================================
from chatgpt_service import load_env, answer_from_chatgpt, doc_to_chroma
from dotenv import load_dotenv, find_dotenv
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
import streamlit as st
from pathlib import Path

# 환경변수를 로딩한다.
load_env()

# ----------------------------------------------------------------------------------------------------------------------
# 메인 화면을 초기화 한다.
# ----------------------------------------------------------------------------------------------------------------------
def init_page():
    st.set_page_config(
        page_title="Chatbot Demo"
    )
    st.header("Chatbot Demo using GPT-3.5")
    st.sidebar.title("Options")


# ----------------------------------------------------------------------------------------------------------------------
# 메시지를 초기화 한다.
# ----------------------------------------------------------------------------------------------------------------------
def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(
                content="You are a helpful AI assistant. Respond your answer in mardkown format.")
        ]

def select_model():
    # 환경변수 값들을 읽어와서 보여줄 수 있었으면 좋겠음
    model_name = st.sidebar.radio("Choose LLM:",
                                  ("gpt-3.5-turbo", "gpt-4"))
    temperature = st.sidebar.slider("Temperature:", min_value=0.0,
                                    max_value=1.0, value=0.5, step=0.01)
    return model_name, temperature


# ----------------------------------------------------------------------------------------------------------------------
# 챗GPT API를 호출하는 함수
# ----------------------------------------------------------------------------------------------------------------------
def request_chat_api(user_message: str) -> str:
    answer = answer_from_chatgpt(user_message)

    return answer

# ----------------------------------------------------------------------------------------------------------------------
# 메인 함수
# ----------------------------------------------------------------------------------------------------------------------
def main():
    # _ = load_dotenv(find_dotenv())

    init_page()
    model_name, temperature = select_model()
    init_messages()

    # 사이드바에 파일 업로드 버튼을 생성한다.
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF Files",
        accept_multiple_files=True,
        type=["pdf"]
    )

    # 업로드한 파일들을 ./pdf 디렉토리로 저장한다.
    if uploaded_files:
        pdf_directory = Path("./pdf")
        pdf_directory.mkdir(parents=True, exist_ok=True)

        for file in uploaded_files:
            file_path = pdf_directory / file.name
            with open(file_path, "wb") as f:
                f.write(file.read())

        doc_to_chroma(pdf_directory)

    # 질문이 입력되었는지 확인한다.
    if user_input := st.chat_input("Input your question!"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            answer = request_chat_api(user_input)
        st.session_state.messages.append(AIMessage(content=answer))

    # 기존 대화내용을 보여준다.
    messages = st.session_state.get("messages", [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.markdown(message.content)


# ----------------------------------------------------------------------------------------------------------------------
# 메인함수 호출
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    main()



