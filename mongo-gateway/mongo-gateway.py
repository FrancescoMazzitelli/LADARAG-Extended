from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
from bson.json_util import dumps
from cheroot.wsgi import Server as WSGIServer
import os

# Embedding model import
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

MONGO_USER = os.environ.get("MONGO_USER", "admin")
MONGO_PASS = os.environ.get("MONGO_PASS", "admin")
MONGO_HOST = os.environ.get("MONGO_HOST", "catalog")
MONGO_PORT = os.environ.get("MONGO_PORT", "27017")

MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/"
MONGO_DB = os.environ.get("MONGO_DB", "microcks")
client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db["services"]

embedding_model = SentenceTransformer(model_name_or_path='intfloat/multilingual-e5-large', device='cpu', trust_remote_code=True)

def clean_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc

def embed(input):
        embedding = embedding_model.encode(str(input), convert_to_tensor=True, normalize_embeddings=True)
        return [float(x) for x in embedding]

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Mongo Server is running"}), 200

@app.route("/index/create", methods=["POST"])
def create_vector_index():
    collection.create_search_index(
        {
            "name": "vector_index",
            "type": "vectorSearch",
            "definition": {
                "fields": [
                    {
                        "type": "vector",
                        "path": "embedding",
                        "similarity": "cosine",
                        "numDimensions": 1024
                    }
                ]
            }
        }
    )

    print("🔍 Created vector index on 'embedding' field")

@app.route("/index/search", methods=["POST"])
def vector_search():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query_text = data["query"]
    query_embedding = embed(query_text)

    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embedding,
                "path": "embedding",
                "exact": True,
                "limit": 5
            }
        },
        {
            "$addFields": {
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]


    results = list(collection.aggregate(pipeline))
    for doc in results:
        doc.pop("embedding", None)
        clean_doc(doc)
    return jsonify({"results": results}), 200


@app.route("/service", methods=["POST"])
def create_or_update_service():
    data = request.get_json()
    if not data or "id" not in data:
        return jsonify({"error": "Missing 'id' field"}), 400

    doc_id = data["id"]
    data["_id"] = doc_id
    data.pop("id", None)

    description = data.get("description", "")
    capabilities = " ".join(data.get("capabilities", [])) if isinstance(data.get("capabilities"), list) else str(data.get("capabilities", ""))
    text_to_embed = f"{description} {capabilities}".strip()

    embedding = embed(text_to_embed)
    data["embedding"] = embedding

    collection.replace_one({"_id": doc_id}, data, upsert=True)
    return jsonify({"status": "ok", "id": doc_id}), 200


@app.route("/services", methods=["GET"])
def list_services():
    docs = list(collection.find())
    return dumps(docs), 200 

@app.route("/services/<string:service_id>", methods=["GET"])
def get_service(service_id):
    doc = collection.find_one({"_id": service_id})
    if not doc:
        return jsonify({"error": "Service not found"}), 404
    return dumps(doc), 200

@app.route("/services/<string:service_id>", methods=["DELETE"])
def delete_service(service_id):
    result = collection.delete_one({"_id": service_id})
    if result.deleted_count == 0:
        return jsonify({"error": "Service not found"}), 404
    return jsonify({"status": "deleted", "id": service_id}), 200

if __name__ == "__main__":
    server = WSGIServer(('0.0.0.0', 5000), app)
    try:
        print("🚀 Starting Flask app with Cheroot on http://0.0.0.0:5000")
        server.start()
    except KeyboardInterrupt:
        print("🛑 Shutting down server...")
        server.stop()
