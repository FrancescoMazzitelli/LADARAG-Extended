from flask import Flask, request, jsonify
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from bson import ObjectId
from bson.json_util import dumps
from cheroot.wsgi import Server as WSGIServer
import multiprocessing
import uuid
import os
import sys
import logging

from sentence_transformers import SentenceTransformer

log_file_path = "test.txt"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("app")

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

is_server_ready = False

def clean_doc(doc):
    doc["_id"] = str(doc["_id"])
    return doc

def embed(input):
        embedding = embedding_model.encode(f"query: {input}", convert_to_tensor=False, normalize_embeddings=True)
        return embedding.tolist()

# ------------------------------------------------------------------------------| parallel
def init_model():
    global model
    model = SentenceTransformer('intfloat/multilingual-e5-large', device='cpu')

def embed_item(args):
    doc_id, key, text = args
    vector = model.encode(f"query: {text}", normalize_embeddings=True)
    vector_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, text))
    return PointStruct(
        id=vector_id,
        vector=vector.tolist(),
        payload={"mongo_id": doc_id, "http_operation": key}
    )
# ------------------------------------------------------------------------------| parallel
def create_vector_collection():
    qdrant_client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
    )
    return jsonify({"status": "Collection created"}), 200

@app.route("/health")
def index():
    if is_server_ready is True:
        return jsonify({"status": "ok", "message": "Gateway Server is ready", "model_loaded": True}), 200
    else:
        logger.error(f"Model not yet loaded or broken")
        return jsonify({"status": "error", "message": "Model not yet loaded or broken", "model_loaded": False}), 500
        


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


@app.route("/service/old", methods=["POST"])
def create_or_update_service_old():
    data = request.get_json()
    if not data or "id" not in data:
        return jsonify({"error": "Missing 'id' field"}), 400

    doc_id = data["id"]
    data["_id"] = doc_id
    data.pop("id", None)

    capabilities = data.get("capabilities")
    for k, v in capabilities.items():
        embedding = embed(v)
        print(f"EMBEDDING DIM: {len(embedding)}")
        vector_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, v))
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=vector_id,
                    vector=embedding,
                    payload={"mongo_id": doc_id, "http_operation": k}
                )
            ]
        )

    collection.replace_one({"_id": doc_id}, data, upsert=True)
    return jsonify({"status": "ok", "id": doc_id}), 200

# ------------------------------------------------------------------------------| parallel
@app.route("/service", methods=["POST"])
def create_or_update_service():
    data = request.get_json()
    if not data or "id" not in data:
        return jsonify({"error": "Missing 'id' field"}), 400

    doc_id = data["id"]
    data["_id"] = doc_id
    data.pop("id", None)
    
    capabilities = data.get("capabilities")
    input_data = [(doc_id, k, v) for k, v in capabilities.items()]

    try:
        with multiprocessing.Pool(initializer=init_model) as pool:
            points = pool.map(embed_item, input_data)
    except Exception as e:
        logger.exception("Embedding failed")
        return jsonify({"error": "Embedding failed", "details": str(e)}), 500

    qdrant_client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points
    )

    collection.replace_one({"_id": doc_id}, data, upsert=True)
    return jsonify({"status": "ok", "id": doc_id}), 200
# ------------------------------------------------------------------------------| parallel

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
    is_server_ready = True
    try:
        print("🚀 Starting Flask app with Cheroot on http://0.0.0.0:5000")
        server.start()
    except KeyboardInterrupt:
        print("🛑 Shutting down server...")
        server.stop()
