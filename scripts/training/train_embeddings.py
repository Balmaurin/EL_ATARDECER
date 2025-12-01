
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle

def train_embeddings():
    # Cargar datos
    with open("data/training_data.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Preparar textos
    texts = []
    for item in data:
        texts.append(item["input"])
        texts.append(item["output"])
    
    # Crear vectorizador TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Reducir dimensiones con SVD
    svd = TruncatedSVD(n_components=100)
    embeddings = svd.fit_transform(tfidf_matrix)
    
    # Guardar modelos
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    
    with open("models/svd.pkl", "wb") as f:
        pickle.dump(svd, f)
    
    np.save("models/embeddings.npy", embeddings)
    
    print(f"Embeddings entrenados: {embeddings.shape}")
    return True

if __name__ == "__main__":
    train_embeddings()
