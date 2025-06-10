import torch
import pickle
import numpy as np
from sentence_transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import rag_samba_continuous_function as rag

default_pdf_path = "/home/sagar/Master_pdfs/pdfs/"
default_encode_path = "/home/sagar/Master_pdfs/encodings/"
default_chunks_path = "/home/sagar/Master_pdfs/chunks/"

unique_laptop = {'1':'1.pdf'}

for pdf_file in unique_laptop.values():
    if pdf_file != "Not Found":                  
        pdf_file_path = default_pdf_path + pdf_file
        chunks = rag.get_chunks(pdf_file_path)

    # Ensure device is set correctly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context_encoder = DPRContextEncoder.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
    ).to(device)
    context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L12-v2"
    )

    # Tokenize and encode context chunks
    context_encodings = []
    for chunk in chunks:
        inputs = context_tokenizer(
            chunk, return_tensors="pt", truncation=True, max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            context_encoding = context_encoder(**inputs).pooler_output
        context_encodings.append(context_encoding.cpu().numpy())

    context_encodings = np.vstack(context_encodings)
    default_encode_path = "/home/sagar/Master_pdfs/encodings/"
    # Save encodings to a file

    encodings_filename = default_encode_path + f"{pdf_file.split('.')[0]}.npy"
    np.save(encodings_filename, context_encodings)

    default_chunks_path = "/home/sagar/Master_pdfs/chunks/"
    chunks_filename = default_chunks_path + f"{pdf_file.split('.')[0]}.pkl"
    with open(chunks_filename, 'wb') as f:
        pickle.dump(chunks, f)