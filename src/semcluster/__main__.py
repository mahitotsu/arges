import boto3
import json 
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN

bedrock_client = boto3.client(
    service_name="bedrock-runtime", 
    region_name="ap-northeast-1"
) 

def get_embedding(text: str) -> list[float]:
    response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text}),
        accept="application/json",
        contentType="application/json",
    )
    response_body = json.loads(response.get("body").read().decode())
    return response_body.get('embedding')

def main():
    filepath = Path('./data/simple.csv').resolve()
    df = pd.read_csv(filepath)
    df['embedding'] = df['description'].apply(get_embedding)

    vectors = np.array(df['embedding'].to_list())
    print(vectors)
    model = DBSCAN(eps=0.5, min_samples=2)
    labels = model.fit_predict(vectors)
    print(labels)

if __name__ == "__main__":
    main()