import requests
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"
OLLAMA_EMBED_URL = "http://localhost:11434/api/embeddings"
LLM_MODEL = "gemma4:cloud" 
EMBED_MODEL = "mxbai-embed-large"

# Initialize ChromaDB (Persistent storage in a folder called 'my_vector_db')
db_client = chromadb.PersistentClient(path="./my_vector_db")
collection = db_client.get_or_create_collection(name="document_collection")

def get_embedding(text):
    """
    Converts a string of text into a vector (list of floats) using Ollama.
    """
    payload = {"model": EMBED_MODEL, "prompt": text}
    response = requests.post(OLLAMA_EMBED_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    embedding = data.get("embedding")
    if not embedding:
        raise RuntimeError(
            f"Ollama returned no embedding for model '{EMBED_MODEL}'. "
            f"Response: {data}. "
            f"Make sure the model is pulled: `ollama pull {EMBED_MODEL}`"
        )
    return embedding

def ingest_document(file_path):
    """
    Reads a document, splits it into chunks, embeds them, and stores them in ChromaDB.
    """
    print(f"📄 Processing {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # 1. Split text into manageable chunks (500 characters with some overlap)
    # This ensures the model doesn't lose context between chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50
    )
    chunks = text_splitter.split_text(text)
    
    # 2. Embed and store each chunk
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            ids=[f"chunk_{i}"], 
            embeddings=[embedding], 
            documents=[chunk]
        )
    print(f"✅ Successfully indexed {len(chunks)} chunks.")

def rag_query(user_query):
    """
    Retrieves relevant chunks and generates an answer using the LLM.
    """
    # 1. Embed the user's question
    query_embedding = get_embedding(user_query)

    # 2. Search Vector DB for the top 3 most relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=3
    )
    
    # Extract the text of the retrieved chunks
    relevant_context = "\n\n".join(results['documents'][0])

    # 3. Build the augmented prompt
    prompt = f"""
    You are a helpful assistant. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer based on the context, just say that you don't know.

    CONTEXT:
    {relevant_context}

    QUESTION: 
    {user_query}

    ANSWER:
    """

    # 4. Generate response from Gemma
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }
    
    print(f"\n❓ payload: {payload}")

    response = requests.post(OLLAMA_GENERATE_URL, json=payload)
    return response.json().get("response")

# --- Main Execution ---
if __name__ == "__main__":
    # Create a dummy large file for demonstration
    doc_name = "knowledge_base.txt"
    with open(doc_name, "w") as f:
        f.write("The company 'NebulaCorp' was founded in 2024. " * 10) # Noise
        f.write("\nNebulaCorp's primary product is the Quantum-Sponge, which absorbs dark matter. ")
        f.write("\nTheir CEO is a cat named Barnaby. ")
        f.write("\nThe headquarters are located in a floating city above Jupiter. ")
        f.write("\nThey offer dental insurance but not vision insurance.")

    # Step 1: Ingest the data (You only need to do this once!)
    ingest_document(doc_name)

    # Step 2: Ask a question
    question = "Who is the CEO of NebulaCorp and where is the headquarters?"
    print(f"\n❓ Question: {question}")
    
    answer = rag_query(question)
    print(f"\n🤖 AI Answer:\n{answer}")
