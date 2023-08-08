import base64
import google.auth
import google.auth.transport.requests
import json
import os
import requests

from flask import Flask, jsonify, request
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

app = Flask(__name__)

PROJECT_ID = "YOUR PROJECT HERE"

creds, project = google.auth.default()

auth_req = google.auth.transport.requests.Request()
creds.refresh(auth_req)

def initialize():
    global data
    with open('data.json', 'r') as f:
        data = json.load(f)

    embeddings = [item['embedding'] for item in data['embeddings']]

    similarity_matrix = cosine_similarity(embeddings, embeddings)
    # similarity_matrix = euclidean_distances(embeddings, embeddings)

    cluster_model = AgglomerativeClustering(n_clusters=None, distance_threshold=8.0).fit(similarity_matrix)
    # cluster_model = KMeans(n_clusters=8).fit(similarity_matrix)

    for i in range(len(cluster_model.labels_)):
        data['embeddings'][i]['cluster'] = int(cluster_model.labels_[i])
        local_filename = data['embeddings'][i]["image"].replace(
            "gs://YOUR BUCKET HERE/JaneGoodall/",
            "static/images/"
        )
        with open(local_filename, 'rb') as image_file:
            data['embeddings'][i]["base64image"] = base64.b64encode(image_file.read()).decode('utf-8')


@app.route('/')
def home():
    return app.send_static_file('index.html')

@app.route('/api/entities', methods=['GET'])
def entities():
    return jsonify({
        "embeddings": [
            {
                "image": embedding["image"],
                "cluster": embedding["cluster"]
            }
            for embedding in data['embeddings']
        ]
    })

@app.route('/api/explain', methods=['POST'])
def explain():
    # Get the cluster specification from the request
    cluster = int(request.json['cluster'].replace("Cluster ", ""))
    prompt = request.json['prompt']

    # Explain each frame, then do an aggregate explaination
    frame_explanations = []
    for embedding in data['embeddings']:
        if embedding["cluster"] == cluster:
            response = requests.post(
                f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/imagetext:predict",
                headers={
                    "Authorization": f"Bearer {creds.token}"
                },
                json={
                    "instances": [
                        {
                            "prompt": prompt,
                            "image": {"bytesBase64Encoded": embedding["base64image"]}
                        }
                    ]
                }
            )
            if response.status_code != 200:
                print(response.status_code)
                print(response.json())
                return jsonify(response.json()), response.status_code

            frame_explanations.append(response.json()["predictions"][0])

    frame_explanations_list = "\n".join([f"Frame {i+1}: {explanation}" for i, explanation in enumerate(frame_explanations)])
    response = requests.post(
        f"https://us-central1-aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/us-central1/publishers/google/models/text-bison:predict",
        headers={
            "Authorization": f"Bearer {creds.token}"
        },
        json={
            "instances": [
                { "prompt": f"\n{frame_explanations_list}\nAnswer the question \"{prompt}\" based on summarizing the preceeding frame answers." }
            ],
            "parameters": {
                "temperature": 0.2,
                "maxOutputTokens": 256,
                "topK": 40,
                "topP": 0.95
            }
        }
    )
    if response.status_code != 200:
        print(response.status_code)
        print(response.json())
        return jsonify(response.json()), response.status_code

    return jsonify({
        "message": response.json()["predictions"][0]["content"],
        "frames": frame_explanations
    })


if __name__ == '__main__':
    initialize()
    app.run(debug=True, host='0.0.0.0', port=8080)