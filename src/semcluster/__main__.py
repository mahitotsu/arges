import boto3
import json 
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Sequence, Union
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer

bedrock_client = boto3.client(
    service_name="bedrock-runtime", 
    region_name="ap-northeast-1"
) 

# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
model = SentenceTransformer("stsb-xlm-r-multilingual")

def get_embedding(text: str) -> np.ndarray:
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text}),
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read().decode())
    embedding = response_body.get('embedding')
    return np.array(embedding)

def transform(
    texts: Union[str, Sequence[str]],
    *,
    batch_size: int = 32,
    normalize: bool = False,
) -> np.ndarray:
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=normalize,
        show_progress_bar=False,
    )

def main():
    filepath = Path('./data/sample.csv').resolve()
    df = pd.read_csv(filepath)

    descriptions = df['description'].astype(str).tolist()
    embeddings = transform(descriptions)
    df['embedding'] = list(embeddings)

    vectors = embeddings 
    clusterer = DBSCAN(eps=0.5, min_samples=2)
    labels = clusterer.fit_predict(vectors)
    print(labels)

if __name__ == "__main__":
    main()