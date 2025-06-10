from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """Initialize and return HuggingFace embeddings with optimized CPU support"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            'device': 'cpu'
        },
        encode_kwargs={
            'normalize_embeddings': True,  # Normalize embeddings for better similarity search
            'batch_size': 32              # Batch size for encoding
        }
    ) 