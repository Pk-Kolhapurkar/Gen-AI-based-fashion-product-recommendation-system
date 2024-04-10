import os
import pickle
import torch
import pickle
import matplotlib.pyplot as plt
from langchain_community.document_loaders import TextLoader
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import BertModel, BertTokenizer
from langchain_core.prompts import PromptTemplate

os.environ['HUGGINGFACEHUB_API_TOKEN'] = "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"

class Chatbot:
    def __init__(self):
        self.load_data()
        self.load_models()
        self.load_embeddings()
        self.load_template()

    def load_data(self):
        self.data = load_dataset("ashraq/fashion-product-images-small", split="train")
        self.images = self.data["image"]
        self.product_frame = self.data.remove_columns("image").to_pandas()
        self.product_data = self.product_frame.reset_index(drop=True).to_dict(orient='index')

    def load_template(self):
        self.template = """
        You are a fashion shopping assistant that wants to convert customers based on the information given.
        Describe season and usage given in the context in your interaction with the customer.
        Use a bullet list when describing each product.
        If user ask general question then answer them accordingly, the question may be like when the store will open, where is your store located.
        Context: {context}
        User question: {question}
        Your response: {response}
        """
        self.prompt = PromptTemplate.from_template(self.template)

    def load_models(self):
        self.model = SentenceTransformer('clip-ViT-B-32')
        self.bert_model_name = "bert-base-uncased"
        self.bert_model = BertModel.from_pretrained(self.bert_model_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)
        self.gpt2_model_name = "gpt2"
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(self.gpt2_model_name)
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.gpt2_model_name)

    def load_embeddings(self):
        if os.path.exists("embeddings_cache.pkl"):
            with open("embeddings_cache.pkl", "rb") as f:
                embeddings_cache = pickle.load(f)
            self.image_embeddings = embeddings_cache["image_embeddings"]
            self.text_embeddings = embeddings_cache["text_embeddings"]
        else:
            self.image_embeddings = self.model.encode([image for image in self.images])
            self.text_embeddings = self.model.encode(self.product_frame['productDisplayName'])
            embeddings_cache = {"image_embeddings": self.image_embeddings, "text_embeddings": self.text_embeddings}
            with open("embeddings_cache.pkl", "wb") as f:
                pickle.dump(embeddings_cache, f)

    def create_docs(self, results):
        docs = []
        for result in results:
            pid = result['corpus_id']
            score = result['score']
            result_string = ''
            result_string += "Product Name:" + self.product_data[pid]['productDisplayName'] + \
                             ';' + "Category:" + self.product_data[pid]['masterCategory'] + \
                             ';' + "Article Type:" + self.product_data[pid]['articleType'] + \
                             ';' + "Usage:" + self.product_data[pid]['usage'] + \
                             ';' + "Season:" + self.product_data[pid]['season'] + \
                             ';' + "Gender:" + self.product_data[pid]['gender']
            # Assuming text is imported from somewhere else
            doc = text(page_content=result_string)
            doc.metadata['pid'] = str(pid)
            doc.metadata['score'] = score
            docs.append(doc)
        return docs

    def get_results(self, query, embeddings, top_k=10):
        query_embedding = self.model.encode([query])
        cos_scores = util.pytorch_cos_sim(query_embedding, embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        indices = top_results.indices.tolist()
        scores = top_results.values.tolist()
        results = [{'corpus_id': idx, 'score': score} for idx, score in zip(indices, scores)]
        return results

    def display_text_and_images(self, results_text):
        for result in results_text:
            pid = result['corpus_id']
            product_info = self.product_data[pid]
            print("Product Name:", product_info['productDisplayName'])
            print("Category:", product_info['masterCategory'])
            print("Article Type:", product_info['articleType'])
            print("Usage:", product_info['usage'])
            print("Season:", product_info['season'])
            print("Gender:", product_info['gender'])
            print("Score:", result['score'])
            plt.imshow(self.images[pid])
            plt.axis('off')
            plt.show()

    @staticmethod
    def cos_sim(a, b):
        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm.T, b_norm)  # Reshape a_norm to (768, 1)

    def generate_response(self, query):
        # Process the user query and generate a response
        results_text = self.get_results(query, self.text_embeddings)

        # Generate chatbot response
        chatbot_response = "This is a placeholder response from the chatbot."  # Placeholder, replace with actual response

        # Display recommended products
        self.display_text_and_images(results_text)

        # Return both chatbot response and recommended products
        return chatbot_response,results_text

