import torch
from transformers import BertModel, BertTokenizer  

class BERTCustomEmbeddings:  
    def __init__(self, device="cpu"):
        self.device = torch.device(device)
        self.model = BertModel.from_pretrained(      
            "bert-base-uncased"
        ).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(  
            "bert-base-uncased"
        )

    def embed_documents(self, texts):
        """Return a list of embeddings for the input text chunks"""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}  

            with torch.no_grad():
                output = self.model(**inputs)
                cls_embedding = output.last_hidden_state[:, 0, :]  
            embeddings.append(cls_embedding.cpu().numpy()[0])
        return embeddings


def get_embeddings():
    return BERTCustomEmbeddings(device="cpu")  
