import torch
import pickle
import numpy as np
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
# from transformers import DPRQuestionEncoderTokenizer, DPRQuestionEncoder
import rag_samba_continuous_function as rag

default_pdf_path = "/home/sagar/Master_pdfs/pdfs/"
default_encode_path = "/home/sagar/Master_pdfs/encodings/"
default_chunks_path = "/home/sagar/Master_pdfs/chunks/"

# ✅ Just add the new PDF here
unique_laptop = {
    'lenovo_l14.pdf': 'lenovo_l14.pdf',
    'lenovo Thinkbook 14':'lenovo_thinkbook_14.pdf'

}

# ✅ Load model & tokenizer once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context_encoder = DPRContextEncoder.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
).to(device)
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
    "facebook/dpr-ctx_encoder-single-nq-base"
)

for name, pdf_file in unique_laptop.items():
    if pdf_file != "Not Found":
        pdf_file_path = default_pdf_path + pdf_file
        chunks = rag.get_chunks(pdf_file_path)

        context_encodings = []
        for chunk in chunks:
            inputs = context_tokenizer(
                chunk, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                context_encoding = context_encoder(**inputs).pooler_output
            context_encodings.append(context_encoding.cpu().numpy())

        # Combine and save encodings
        context_encodings = np.vstack(context_encodings)
        np.save(f"{default_encode_path}{name}.npy", context_encodings)

        # Save chunks
        with open(f"{default_chunks_path}{name}.pkl", 'wb') as f:
            pickle.dump(chunks, f)

        print(f"✅ Done: {pdf_file}")

