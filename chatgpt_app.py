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
#   ├- Langchain_XXXX-X.ipynb : 작성된 모듈을 테스트하는 Jupyter Notebook
#   ├- crud : API 서비스를 위한 모듈 ( 비즈니스 모듈 - 대부분의 작업 대상임)
#
# [ 실행 명령 ]
# > streamlit run chatgpt_app.py
# ----------------------------------------------------------------------------------------------------------------------
# 2023.09.08 - 초기모듈 작성 (채팅형식 UI로 변경)
# ======================================================================================================================
from chatgpt_service import load_env, answer_from_chatgpt
import streamlit as st
from streamlit_chat import message

# 환경변수를 로딩한다.
load_env()

st.header("Chatbot Demo using `Streamlit and GPT-3.5`")
st.sidebar.title('Chat GPT (RAG)')

# Storing GPT-3.5 responses for easy retrieval to show on Chatbot UI in Streamlit session
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

# Storing user responses for easy retrieval to show on Chatbot UI in Streamlit session
if 'past' not in st.session_state:
    st.session_state['past'] = []

# Storing total conversation for easy retrieval to show on Chatbot UI in Streamlit session
if 'total_conversation' not in st.session_state:
    st.session_state['total_conversation'] = [
        {'role': 'system',
         'content': 'Use the following pieces of context to answer the users question.If you don\'t know the answer, just say that you don\'t know, don\'t try to make up an answer'
         }]


def query(user_text):
    last_message = st.session_state.total_conversation[-1]
    message = answer_from_chatgpt(last_message["content"])

    # Adding bot response to the
    st.session_state.total_conversation.append({'role': 'system', 'content': message})
    return message


def get_text():
    input_text = st.text_input("You: ", "Hello, How are you?", key="input")
    return input_text


user_input = get_text()

if user_input:
    output = query(st.session_state.total_conversation.append({'role': 'user', 'content': user_input}))

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:

    for i in range(len(st.session_state['generated']) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')



# # [ 삭제예정 ]
# # ======================================================================================================================
# # Streamlit + LangChain + ChatGPT
# # 출처: https://github.com/jaysooo/chatgpt_streamlit_app
# # ----------------------------------------------------------------------------------------------------------------------
# # 2023.09.07 - 초기모듈 작성
# # ======================================================================================================================
# from chatgpt_service import load_env, answer_from_chatgpt
# from chatgpt_logger import logger
# import streamlit as st
#
# # ----------------------------------------------------------------------------------------------------------------------
# # 입력된 채팅을 GPT에 전달하고, 응답을 반환한다.
# # ----------------------------------------------------------------------------------------------------------------------
# def send_callback():
#     query = st.session_state['query']
#     answer = answer_from_chatgpt(query)
#     st.session_state['answer'] = answer
#
#
# # ----------------------------------------------------------------------------------------------------------------------
# # 화면을 초기화한다.
# # ----------------------------------------------------------------------------------------------------------------------
# def clear_callback():
#      st.session_state.clear()
#
#
# # ----------------------------------------------------------------------------------------------------------------------
# # 사용자 채팅화면을 로딩한다.
# # ----------------------------------------------------------------------------------------------------------------------
# def load_streamlit():
#     #st.sidebar.text_input ='test'
#     title = st.title = '## Chat GPT Simple Application .. :robot_face:'
#     st.sidebar.title('Chat GPT (RAG)')
#
#     st.write(title)
#     st.text_input('prompt:keyboard:', key='query')
#
#     col1,col2,col3 = st.columns([1,1,1])
#     with col1:
#         st.button("send",on_click=send_callback)
#     with col3:
#         st.button("clear",on_click=clear_callback)
#
#     st.text_area('response',key='answer',height=500)
#
#
# # ----------------------------------------------------------------------------------------------------------------------
# # 메인 함수
# # ----------------------------------------------------------------------------------------------------------------------
# def main():
#     logger.info("main application start..")
#     load_env()
#     load_streamlit()
#     #answer_from_chatgpt("지금 시간은?")
#
# if __name__=='__main__':
#     main()


