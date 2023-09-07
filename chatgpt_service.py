# ======================================================================================================================
# 챗봇 서비스 모듈
# 출처: https://github.com/jaysooo/chatgpt_streamlit_app
# ----------------------------------------------------------------------------------------------------------------------
# 2023.09.07 - 초기모듈 작성
# ======================================================================================================================
from dotenv import load_dotenv
from chatgpt_logger import logger
import openai
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 환경 변수를 가져온다.
def get_openai_options():
    openai_model = os.environ.get("OPENAI_MODEL")
    openai_temperature = os.environ.get("OPENAI_TEMPERATURE")
    oepnai_max_token =os.environ.get("OPENAI_MAX_TOKEN") 

    args = {
        'model': openai_model,
        'temperature' : openai_temperature,
        'max_token' : oepnai_max_token,
    }

    return args

def load_env():

    # set environment for application
    load_dotenv()
    version = os.environ.get("VERSION")
    openai_token = os.environ.get("OPENAI_TOKEN")
    huggingfacehub_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    version = os.environ.get("VERSION")

    os.environ["OPENAI_API_KEY"] = openai_token
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingfacehub_token
    # os.environ["SERPAPI_API_KEY"] = SERPAPI_API_KEY

    # set openai connection
    openai.api_key=openai_token

    logger.info(f"app version :  {version} \t")


# # ----------------------------------------------------------------------------------------------------------------------
# # PDF 파일을 읽어오기
# # ----------------------------------------------------------------------------------------------------------------------
# from langchain.document_loaders import PyPDFLoader
#
# pdf_dir = "./pdf"
# pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
#
# loaders = []
# for pdf_file in pdf_files:
#     print(pdf_file)
#     loader = PyPDFLoader(pdf_file)
#     loaders.append(loader)
#
# # ----------------------------------------------------------------------------------------------------------------------
# # Embeddings and VectorStore
# # ----------------------------------------------------------------------------------------------------------------------
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.embeddings import OpenAIEmbeddings
#
# # embeddings = OpenAIEmbeddings()
# embeddings = HuggingFaceEmbeddings()
#
# from langchain.chat_models import ChatOpenAI
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.vectorstores import FAISS

# from langchain.text_splitter import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# index = VectorstoreIndexCreator(
#     vectorstore_cls=FAISS,
#     embedding=embeddings,
#     # text_splitter=text_splitter,
#     ).from_loaders([loader])
#
# # 파일로 저장
# index.vectorstore.save_local("faiss-nj")
#
# #@title FAISS 벡터DB 디스크에서 불러오기
# from langchain.indexes.vectorstore import VectorStoreIndexWrapper
#
# fdb = FAISS.load_local("faiss-nj", embeddings)

# ----------------------------------------------------------------------------------------------------------------------
# 챗봇 서비스 API
# ----------------------------------------------------------------------------------------------------------------------
def answer_from_chatgpt(query):
    chat = ChatOpenAI(temperature=0.3)
    index = VectorStoreIndexWrapper(vectorstore=fdb)
    return index.query(query, llm=chat, verbose=True)


# def answer_from_chatgpt(query):
#     #query = 'yarn cluster manager의 개념을 알려줘'
#     answer = ''
#     if query is None or len(query) < 1:
#         answer = 'No Response..'
#         return answer
#
#
#     options = get_openai_options()
#     print("model:", options['model'])
#     # response = openai.Completion.create(model=options['model'], prompt=query, temperature=float(options['temperature']),max_tokens= int(options['max_token']))
#     # res = response['choices'][0]['text']
# #    print(res)
# #     answer = res
#     # for line in res.split("\n"):
#     #     #print(line)
#     #     answer = line
#
#     response = openai.ChatCompletion.create(
#         model='gpt-3.5-turbo',
#         messages=[{"role": "user", "content": query}
#                   ],
#         temperature=float(options['temperature']),
#     )
#     return response.choices[0].message.content