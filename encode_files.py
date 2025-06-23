import os
import json
import torch
import pickle
import numpy as np
from transformers import BertModel, BertTokenizer  

import rag_samba_continuous_function as rag

default_pdf_path = "/home/sagar/Master_pdfs/pdfs/"
default_encode_path = "/home/sagar/Master_pdfs/encodings/"
default_chunks_path = "/home/sagar/Master_pdfs/chunks/"
mapping_file = "pdf_mappings.json"

# Load mapping
with open(mapping_file, 'r') as f:
    unique_laptop = json.load(f)

# BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model = BertModel.from_pretrained("bert-base-uncased").to(device)
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Loop through mapping
for name, pdf_file in unique_laptop.items():
    pdf_path = os.path.join(default_pdf_path, pdf_file)
    encode_file = os.path.join(default_encode_path, f"{pdf_file.split('.')[0]}.npy")
    chunk_file = os.path.join(default_chunks_path, f"{pdf_file.split('.')[0]}.pkl")

    # Skip if already processed
    if os.path.exists(encode_file) and os.path.exists(chunk_file):
        print(f" Skipped (Already encoded): {pdf_file}")
        continue

    # Process PDF
    print(f" Encoding: {pdf_file}")
    chunks = rag.get_chunks(pdf_path)

    context_encodings = []
    for chunk in chunks:
        inputs = bert_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            output = bert_model(**inputs)
            cls_embedding = output.last_hidden_state[:, 0, :]
        context_encodings.append(cls_embedding.cpu().numpy())

    context_encodings = np.vstack(context_encodings)
    np.save(encode_file, context_encodings)

    with open(chunk_file, 'wb') as f:
        pickle.dump(chunks, f)

    print(f" Done: {pdf_file}")
