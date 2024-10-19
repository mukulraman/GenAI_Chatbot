import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain

# Reading the file
def get_llm(path):
    loader=DirectoryLoader(path)
    documents=loader.load()
    
    # Split the file
    text_splitter=CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    texts=text_splitter.split_documents(documents)

    # Enable below line if Multiple Copies of OpenMP are linked to the program
    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    openai_api_key = os.getenv("OPENAI_API_KEY")
    # Vector conversion
    embeddings=OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Finding similar Vector
    vector_db=FAISS.from_documents(texts,embeddings)

    # Creating Prompt and retrive data. Question and relevant information is sent in the form of prompt to llm. This operation is called chain_type=stuff
    llm=OpenAI(openai_api_key=openai_api_key)
    qa=ConversationalRetrievalChain.from_llm(llm=llm,chain_type="stuff",retriever=vector_db.as_retriever())
       
    return qa