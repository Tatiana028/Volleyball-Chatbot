from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI   
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import os
from src.prompt import system_prompt  
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found! Check .env file.")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found! Check .env file")

genai.configure(api_key=GOOGLE_API_KEY)

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()

index_name = "testbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.4)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("index.html")   

@app.route("/get", methods=["POST"])
def chat():
    msg = request.json.get("msg")   
    if not msg:
        return jsonify({"error": "No message provided"}), 400

    response = rag_chain.invoke({"input": msg})
    return jsonify({"response": response["answer"]})

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)