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

# ----------------------------------------------------------------------------------------------------------------------
# 환경 변수들을 딕셔너리로 묶어서 반환한다.
# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
# 환경 변수를 로딩한다.
# ----------------------------------------------------------------------------------------------------------------------
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

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # set openai connection
    openai.api_key = openai_token

    logger.info(f"app version :  {version} \t")

# ----------------------------------------------------------------------------------------------------------------------
# 해당 디렉토리의 인베팅된 문서들로부터 Chroma DB로 생성한다.
# ----------------------------------------------------------------------------------------------------------------------
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings()
persist_directory = "./chroma_db"
# 2023.09.07 - (임시조치) 아래와 같이 try except 처리를 해주지 않으면, 아래와 같은 에러가 발생한다.
#            - 이유는 아직 모르겠고, Chroma DB가 사용하는 의존성 모듈들의 버전이 맞지 않는 것 같음
# InvalidInputException: Invalid Input Error: Required module 'pandas.core.arrays.arrow.dtype' failed to import,
# due to the following Python exception: ModuleNotFoundError: No module named 'pandas.core.arrays.arrow.dtype'
try:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
except:
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)


# ----------------------------------------------------------------------------------------------------------------------
# 챗봇 서비스 API
# ----------------------------------------------------------------------------------------------------------------------
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
def answer_from_chatgpt(query):
    model_name = "gpt-3.5-turbo"
    chat = ChatOpenAI(temperature=0.3)
    llm = ChatOpenAI(model_name=model_name)
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    matching_docs = db.similarity_search(query)

    answer = chain.run(input_documents=matching_docs, question=query)

    return answer

# ----------------------------------------------------------------------------------------------------------------------
# 문서들을 인베팅벡터로 변환하고, Chroma DB를 생성한 후 저장한다.
# - 서비스가 시작될 때마다 문서변환을 하지 않고, 미리 변환된 문서를 로딩하여 사용할 수 있도록 함수를 분리함
# 출처: https://blog.futuresmart.ai/using-langchain-and-open-source-vector-db-chroma-for-semantic-search-with-openais-llm
# > conda install -c conda-forge chromadb
# ----------------------------------------------------------------------------------------------------------------------
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
def doc_to_chroma(directory):
    # 해당 디렉토리의 모든 pdf 파일을 읽어서, 문서를 로딩한다.
    # directory = './pdf'
    loader = DirectoryLoader(directory)
    documents = loader.load()
    # len(documents)

    # 문서를 적당한 크기로 쪼개서, 문서를 분할한다.
    chunk_size = 1000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    # print(len(docs))

    # # 문서를 벡터로 변환하고, Chroma DB 객체를 생성한다.
    # # embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    # embeddings = HuggingFaceEmbeddings()
    # try:
    #     db = Chroma.from_documents(docs, embeddings)
    # except:
    #     db = Chroma.from_documents(docs, embeddings)

    # Chroma DB를 해당 디렉토리에 저장한다.
    persist_directory = "chroma_db"
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embeddings, persist_directory=persist_directory
    )
    vectordb.persist()
