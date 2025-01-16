import bs4
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model = "gpt-3.5-turbo")

loader = WebBaseLoader(
    web_paths=("http://lilianweng.github.io/posts/2023-06-23-agent/"),
    bs_kwargs = dict(
        parse_only = bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")

        )
    )
)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splits= text_splitter.split_documents(docs)
vectorestore = Chroma.from_documents(documents=splits,embedding=OpenAIEmbeddings())

retriever =vectorestore.as_retriever()
#rag-prompt
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
prompt=hub.pull("rlm/rag-promt")
rag_chain=(
    {"context":retriever|format_docs,"question": RunnablePassthrough()}
    |prompt
    |llm
    |StrOutputParser()
)






for chunk in rag_chain.stream("what is maximum inner product search"):
    print(chunk,end="",flush=True)


