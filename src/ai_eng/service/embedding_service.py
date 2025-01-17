"""
This module contains code to use with the Ollama package. 
"""
from typing import List
import ollama
from scipy.spatial import distance

class EmbeddingService:
    """
    Embedding Service
    """
    def __init__(self, model: str):
        self.model = model

    def create_embedding(self, text: str) -> List[float]:
        """
        Create an embedding from the text.
        """
        response = ollama.embeddings(model=self.model, prompt=text)
        return response["embedding"]

    def find_n_closest(
        self,
        query_vector: List[float],
        embeddings: List[List[float]],
        n: int = 3
        )-> list[str]:
        """
        Find the n closest sentences to the text.
        """
        distances = []
        for index, vector in enumerate(embeddings):
            dist = distance.cosine(query_vector, vector)
            distances.append({"index": index, "distance": dist})
        distances_sorted = sorted(distances, key=lambda x: x["distance"])
        return distances_sorted[:n]

if __name__ == '__main__':
    service = EmbeddingService(model="mxbai-embed-large:335m")
    print(service.create_embedding("Hello, world!"))
