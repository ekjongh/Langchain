# ======================================================================================================================
# Streamlit + LangChain + ChatGPT + Chroma DB
# 출처: https://pythonwarriors.com/build-chatbot-using-gpt-4-and-streamlit-in-python/
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
#
# [ 실행 명령 ]
# > streamlit run chatgpt_app.py
# ----------------------------------------------------------------------------------------------------------------------
# 2023.09.08 - 초기모듈 작성 (채팅형식 UI로 변경)
#
# ======================================================================================================================
from chatgpt_service import load_env, answer_from_chatgpt
import streamlit as st
import time
import os

# 환경변수를 로딩한다.
load_env()

# ----------------------------------------------------------------------------------------------------------------------
# 챗GPT API를 호출하는 함수
# ----------------------------------------------------------------------------------------------------------------------
def request_chat_api(user_message: str) -> str:
    resp = answer_from_chatgpt(user_message)

    return resp


# ----------------------------------------------------------------------------------------------------------------------
# 초기화 함수
# ----------------------------------------------------------------------------------------------------------------------
def init_streamlit():
    st.title("Chatbot Demo using GPT-3.5")
    st.sidebar.title('Chat GPT')
    # Add a radio button to the sidebar
    selected_option = st.sidebar.radio("Select a RAG method:", ["Documents(pdf)", "Web Search", "Etc"])
    st.write("You selected:", selected_option)

    # # 사이드바에 파일 업로드 위젯 추가
    # uploaded_file = st.sidebar.file_uploader("Upload a file")
    # if uploaded_file is not None:
    #     file_name = os.path.basename(uploaded_file.name)
    #
    #     destination_dir = './pdf'
    #     destination_path = os.path.join(destination_dir, file_name)
    #
    #     # 선택된 파일을 지정된 디렉토리에 저장한다.
    #     with open(destination_path, 'wb') as f:
    #         f.write(uploaded_file.read())
    #
    #     st.write(f"File '{file_name}' uploaded and saved to './pdf' directory.")


    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


# ----------------------------------------------------------------------------------------------------------------------
# 메인 함수
# ----------------------------------------------------------------------------------------------------------------------
def chat_main():
    if message := st.chat_input(""):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": message})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(message)

        # Display assistant response in chat message container
        assistant_response = request_chat_api(message)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            for lines in assistant_response.split("\n"):
                for chunk in lines.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response)
                full_response += "\n"
            message_placeholder.markdown(full_response)

        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )

# ----------------------------------------------------------------------------------------------------------------------
# 초기화 및 에인함수 호출
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    init_streamlit()
    chat_main()

