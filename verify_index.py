import time
from src.retriever.vector_store import VectorStore
import json

def verify():
    print("Verifying persistent index reload...")
    start = time.time()
    store = VectorStore(persist_directory="data/index", collection_name="adaptive_rag_corpus")
    load_time = time.time() - start
    
    count = store.count()
    print(f"Collection count: {count}")
    print(f"Reload time: {load_time:.2f} seconds")
    
    if count == 0:
        print("ERROR: Collection is empty!")
        return

    # Let's get the first record we inserted to verify metadata
    # From the file, chunk 00160774f5a61f00626e2c85f45aa053_0 is the first.
    # We will just read the first line of chunks.jsonl to get a valid chunk_id
    with open("datasets/corpus/chunks.jsonl", "r", encoding="utf-8") as f:
        first_line = f.readline()
        if first_line:
            record = json.loads(first_line)
            chunk_id = record["chunk_id"]
            
            retrieved = store.get_record(chunk_id)
            if retrieved:
                print("Successfully retrieved representative record.")
                print(f"Chunk ID: {retrieved['chunk_id']}")
                print(f"Document ID: {retrieved['metadata']['document_id']}")
                if 'embedding' in retrieved and retrieved['embedding']:
                    print(f"Embedding dimension: {len(retrieved['embedding'])}")
                else:
                    print("ERROR: Embedding missing from retrieved record!")
            else:
                print(f"ERROR: Could not retrieve chunk_id: {chunk_id}")

if __name__ == "__main__":
    verify()
