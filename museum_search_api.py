import modal
import json
from typing import List, Dict, Any, Optional

# Create the Modal app
app = modal.App("museum-search-api")

# Create image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install(["git"])
    .pip_install([
        "fastapi",
        "torch",
        "torchvision",
        "Pillow",
        "numpy",
        "h5py",
        "requests",
        "ftfy",
        "regex",
        "tqdm"
    ])
    .pip_install("git+https://github.com/openai/CLIP.git")
    # Add your generated embeddings file here
    .add_local_file("embeddings.h5", remote_path="/embeddings.h5")
    .add_local_file("all_data.jsonl", remote_path="/all_data.jsonl")
)

# Global variables to store loaded data
embeddings_data = None
museum_records = None
clip_model = None
preprocess = None

@app.function(
    image=image,
    gpu="T4",
    keep_warm=1,  # Keep one instance warm for faster responses
    memory=8192,
    timeout=3600
)
@modal.web_endpoint(method="POST", label="museum-search")
def search_similar_images(request_data: dict) -> Dict[str, Any]:
    """
    Search for similar museum artifacts based on text query or image URL
    
    Expected request format:
    {
        "query": "painting of fruit",  # text query
        "image_url": "https://...",    # OR image URL
        "top_k": 10                    # number of results to return
    }
    """
    global embeddings_data, museum_records, clip_model, preprocess
    
    # Load data if not already loaded
    if embeddings_data is None:
        print("Loading embeddings and data...")
        load_data()
    
    # Parse request
    text_query = request_data.get("query", "")
    image_url = request_data.get("image_url", "")
    top_k = request_data.get("top_k", 10)
    
    if not text_query and not image_url:
        return {"error": "Please provide either a text query or image URL"}
    
    try:
        # Generate query embedding
        if image_url:
            query_embedding = get_image_embedding(image_url)
            search_type = "image"
        else:
            query_embedding = get_text_embedding(text_query)
            search_type = "text"
        
        if query_embedding is None:
            return {"error": "Failed to generate embedding"}
        
        # Search for similar items
        similar_items = find_similar_items(query_embedding, top_k)
        
        return {
            "query": text_query or image_url,
            "search_type": search_type,
            "results": similar_items,
            "total_results": len(similar_items)
        }
        
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}

def load_data():
    """Load embeddings and museum data"""
    import numpy as np
    import h5py
    import clip
    import torch
    
    global embeddings_data, museum_records, clip_model, preprocess
    
    # Load CLIP model
    print("Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    
    # Load embeddings
    print("Loading embeddings...")
    with h5py.File("/embeddings.h5", "r") as f:
        embeddings = f["embeddings"][:]
        ids = f["ids"][:]
        record_indices = f["record_indices"][:]
        
        # Normalize embeddings
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        embeddings_data = {
            "embeddings": embeddings,
            "ids": [id.decode('utf-8') if isinstance(id, bytes) else str(id) for id in ids],
            "record_indices": record_indices
        }
    
    # Load museum records
    print("Loading museum records...")
    museum_records = {}
    with open("/all_data.jsonl", "r", encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                record = json.loads(line)
                # Use the accession number as key
                accession_number = record.get("info", {}).get("Accession Number", f"record_{i}")
                museum_records[accession_number] = record
    
    print(f"Loaded {len(embeddings_data['embeddings'])} embeddings and {len(museum_records)} records")

def get_text_embedding(text: str) -> Optional[List[float]]:
    """Generate embedding for text query"""
    import clip
    import torch
    import numpy as np
    
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_tokens = clip.tokenize([text], truncate=True).to(device)
        
        with torch.no_grad():
            embedding = clip_model.encode_text(text_tokens).cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()  # Convert to list for JSON serialization
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        return None

def get_image_embedding(image_url: str) -> Optional[List[float]]:
    """Generate embedding for image URL"""
    import torch
    import numpy as np
    import requests
    from PIL import Image
    from io import BytesIO
    
    try:
        # Download image
        response = requests.get(image_url, timeout=10, verify=False,
                              headers={'User-Agent': 'Mozilla/5.0 (compatible; MuseumBot/1.0)'})
        if response.status_code != 200:
            return None
        
        # Process image
        image = Image.open(BytesIO(response.content))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate embedding
        device = "cuda" if torch.cuda.is_available() else "cpu"
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            embedding = clip_model.encode_image(image_tensor).cpu().numpy().flatten()
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()  # Convert to list for JSON serialization
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        return None

def find_similar_items(query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
    """Find similar items using cosine similarity"""
    import numpy as np
    
    # Convert back to numpy array
    query_embedding = np.array(query_embedding)
    
    # Calculate similarities
    similarities = np.dot(embeddings_data["embeddings"], query_embedding)
    
    # Get top k indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        record_id = embeddings_data["ids"][idx]
        similarity = float(similarities[idx])
        
        # Get record details
        if record_id in museum_records:
            record = museum_records[record_id]
            info = record.get("info", {})
            
            result = {
                "id": record_id,
                "similarity": similarity,
                "image_url": record.get("image_url", ""),
                "title": info.get("Title", "Unknown Title"),
                "artist": info.get("Main Artist", "Unknown Artist"),
                "museum": info.get("Museum Name", "Unknown Museum"),
                "description": info.get("Brief Description", ""),
                "object_type": info.get("Object Type", ""),
                "medium": info.get("Medium", ""),
                "dimensions": info.get("Dimensions", "")
            }
            results.append(result)
    
    return results

# Health check endpoint
@app.function(image=image)
@modal.web_endpoint(method="GET", label="museum-health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Museum search API is running"}

# CORS-enabled search endpoint for frontend
@app.function(
    image=image,
    gpu="T4",
    keep_warm=1,
    memory=8192,
    timeout=3600
)
@modal.web_endpoint(
    method="POST", 
    label="museum-search-cors",
    custom_domains=["your-domain.com"] if False else None  # Set to True and add your domain if needed
)
def search_with_cors(request_data: dict) -> Dict[str, Any]:
    """Search endpoint with CORS headers for frontend"""
    # Call the main search function
    result = search_similar_images.local(request_data)
    
    # Add CORS headers (Modal will handle this automatically for web endpoints)
    return result

# Browse all artworks endpoint
@app.function(image=image, memory=4096)
@modal.web_endpoint(method="GET", label="browse-artworks")
def browse_artworks(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    """Browse artworks without search - for gallery view"""
    global museum_records
    
    if museum_records is None:
        load_data()
    
    # Convert to list and apply pagination
    all_records = list(museum_records.values())
    total = len(all_records)
    
    # Apply offset and limit
    paginated_records = all_records[offset:offset + limit]
    
    results = []
    for record in paginated_records:
        info = record.get("info", {})
        result = {
            "id": info.get("Accession Number", "unknown"),
            "image_url": record.get("image_url", ""),
            "title": info.get("Title", "Unknown Title"),
            "artist": info.get("Main Artist", "Unknown Artist"),
            "museum": info.get("Museum Name", "Unknown Museum"),
            "description": info.get("Brief Description", ""),
            "object_type": info.get("Object Type", ""),
            "medium": info.get("Medium", ""),
            "dimensions": info.get("Dimensions", "")
        }
        results.append(result)
    
    return {
        "results": results,
        "total": total,
        "offset": offset,
        "limit": limit,
        "has_more": offset + limit < total
    }