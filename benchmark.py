import faiss

def benchmark_embeddings(embeddings):
    # Index the embeddings using FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Measure retrieval performance
    D, I = index.search(embeddings, 5)  # Search top 5 nearest neighbors
    return D, I
