import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def build_rag(doc_path: str, collection_name: str = "commercial-catalog"):
    # Carrega variáveis de ambiente e obtém a API key do Gemini
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY/GEMINI_API_KEY não configurado. Defina a variável de ambiente ou um .env."
        )

    with open(doc_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.create_documents([raw_text])
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)
    vs = Chroma.from_documents(docs, embedding=embeddings, collection_name=collection_name)
    k = int(os.environ.get("RETRIEVER_K", "4"))
    retriever = vs.as_retriever(k=k)

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = PromptTemplate.from_template(
        "Use os contextos abaixo do catálogo para responder com precisão.\n\n"
        "Contextos:\n{context}\n\nPergunta:\n{question}\n\n"
        "Responda de forma concisa e fiel ao catálogo."
    )
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0, google_api_key=api_key)
    chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnableLambda(lambda x: x)}
        | prompt
        | llm
        | StrOutputParser()
    )
    return retriever, chain, embeddings