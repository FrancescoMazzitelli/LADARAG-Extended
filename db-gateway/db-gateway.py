from flask import Flask, request, jsonify
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from bson import ObjectId
from bson.json_util import dumps
from cheroot.wsgi import Server as WSGIServer
import uuid
import os

from sentence_transformers import SentenceTransformer

app = Flask(__name__)

MONGO_USER = os.environ.get("MONGO_USER", "admin")
MONGO_PASS = os.environ.get("MONGO_PASS", "admin")
MONGO_HOST = os.environ.get("MONGO_HOST", "catalog-data")
MONGO_PORT = os.environ.get("MONGO_PORT", "27017")
MONGO_DB = os.environ.get("MONGO_DB", "microcks")
MONGO_URI = f"mongodb://{MONGO_USER}:{MONGO_PASS}@{MONGO_HOST}:{MONGO_PORT}/"


QDRANT_HOST = os.environ.get("QDRANT_HOST", "catalog-vector")
QDRANT_PORT = os.environ.get("QDRANT_PASS", "6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "services")
QDRANT_URI = f"http://{QDRANT_HOST}:{QDRANT_PORT}"


mongo_client = MongoClient(MONGO_URI)
qdrant_client = QdrantClient(QDRANT_URI)

db = mongo_client[MONGO_DB]
collection = db["services"]

embedding_model = SentenceTransformer(model_name_or_path='intfloat/multilingual-e5-large', device='cpu', trust_remote_code=True)

def clean_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc

def embed(input):
        embedding = embedding_model.encode(str(input), convert_to_tensor=True, normalize_embeddings=True)
        return embedding.tolist()

def create_vector_collection():
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    return jsonify({"status": "Collection created"}), 200

@app.route("/")
def index():
    return jsonify({"status": "ok", "message": "Mongo Server is running"}), 200

@app.route("/index/search", methods=["POST"])
def vector_search():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    query_text = data["query"]
    query_embedding = embed(query_text)

    results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_embedding,
        limit=5
    )

    services = []
    for result in results:
        doc_id = result.payload["mongo_id"]
        service = collection.find_one({"_id": doc_id})
        services.append(service)
    return jsonify({"results": services}), 200


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
    vector_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            PointStruct(
                id=vector_id,
                vector=embedding,
                payload={"mongo_id": doc_id}
            )
        ]
    )

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
    with app.app_context():
        result = create_vector_collection()
        print(result)
    server = WSGIServer(('0.0.0.0', 5000), app)
    try:
        print("🚀 Starting Flask app with Cheroot on http://0.0.0.0:5000")
        server.start()
    except KeyboardInterrupt:
        print("🛑 Shutting down server...")
        server.stop()
