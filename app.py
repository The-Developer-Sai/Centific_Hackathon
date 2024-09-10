from flask import Flask, request, jsonify
from embedding_pipeline import EmbeddingPipeline
from waitress import serve

app = Flask(__name__)
pipeline = EmbeddingPipeline()

@app.route('/api/v1/embed', methods=['POST'])
def embed():
    data = request.json.get('text')
    embeddings = pipeline.embed_text(data)
    return jsonify({"embeddings": embeddings.tolist()})


@app.route('/api/v1/switch-model', methods=['POST'])
def switch_model():
    model_name = request.json.get('model_name')
    try:
        pipeline.switch_model(model_name)
        return jsonify({"message": f"Switched to {model_name}"})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/v1/benchmark', methods=['POST'])
def benchmark():
    model_name = request.json.get('model_name')
    # Add benchmarking logic here (for speed, quality, etc.)
    # Use FAISS or ElasticSearch for retrieval benchmarking
    return jsonify({"benchmark": "Results"})

@app.route('/api/v1/models', methods=['GET'])
def get_models():
    return jsonify({"available_models": list(pipeline.models.keys())})

if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=5000)
