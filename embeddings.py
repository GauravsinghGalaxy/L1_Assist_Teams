# 


import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

class DPRCustomEmbeddings:
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        ).to(self.device)
        self.tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )

    def embed_documents(self, texts):
        """Return a list of embeddings for the input text chunks"""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)
            with torch.no_grad():
                output = self.model(**inputs).pooler_output
            embeddings.append(output.cpu().numpy()[0])
        return embeddings

# Factory method to return the embedding instance
def get_embeddings():
    return DPRCustomEmbeddings(device="cpu")
