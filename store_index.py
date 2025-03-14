
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import pinecone
import os
from pinecone import ServerlessSpec 


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found! Check .env file.")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

extracted_data = load_pdf_file("/Users/tatiana28/Downloads/Volleyball-Chatbot/Data")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()

pc = pinecone.Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "testbot"
 
if index_name not in pc.list_indexes():
    pc.create_index(
        name=index_name,
        dimension=384,   
        metric="cosine",  
        spec=ServerlessSpec(cloud="aws", region="us-east-1")   
    )
 
index = pc.Index(index_name)
 
print(pc.list_indexes())

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)