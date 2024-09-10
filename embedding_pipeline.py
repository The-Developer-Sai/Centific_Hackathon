from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer
import openai
import numpy as np

# Load BERT model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load Sentence Transformer model
sentence_transformer_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# OpenAI API setup (replace with your OpenAI API key)
openai.api_key = 'your_openai_api_key'

def get_bert_embeddings(text):
    inputs = bert_tokenizer(text, return_tensors='pt')
    outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

def get_sentence_transformer_embeddings(text):
    return sentence_transformer_model.encode(text)

def get_openai_embeddings(text):
    response = openai.Embedding.create(input=[text], engine="text-embedding-ada-002")
    return np.array(response['data'][0]['embedding'])

class EmbeddingPipeline:
    def __init__(self):
        self.models = {
            'bert': get_bert_embeddings,
            'sentence_transformer': get_sentence_transformer_embeddings,
            'openai': get_openai_embeddings
        }
        self.active_model = 'bert'

    def switch_model(self, model_name):
        if model_name in self.models:
            self.active_model = model_name
        else:
            raise ValueError("Model not supported")

    def embed_text(self, text):
        return self.models[self.active_model](text)
