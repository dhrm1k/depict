import modal
import json
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple, Any, Dict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import requests
from io import BytesIO
import h5py
import warnings

warnings.filterwarnings('ignore', message='Unverified HTTPS request')

# Define the Modal app
app = modal.App("museum-embeddings")

# Create Modal image with dependencies and local data files
image = (
    modal.Image.debian_slim()
    .apt_install(["git", "wget", "curl"])
    .pip_install([
        "torch",
        "torchvision", 
        "torchaudio",
        "Pillow",
        "numpy",
        "h5py",
        "requests",
        "ftfy",
        "regex",
        "tqdm"
    ])
    .pip_install("git+https://github.com/openai/CLIP.git")
    .add_local_file("ngma_blr.json", remote_path="/ngma_blr.json")
    .add_local_file("nkm_hyd.json", remote_path="/nkm_hyd.json")
    .add_local_file("nat_del.json", remote_path="/nat_del.json")  # Fixed this line
    .add_local_file("im_kol.json", remote_path="/im_kol.json")
    .add_local_file("gom_goa.json", remote_path="/gom_goa.json")
    .add_local_file("alh_ald.json", remote_path="/alh_ald.json")
)

@dataclass
class MuseumRecord:
    accession_number: str
    museum_name: str
    image_url: str
    gallery_name: Optional[str] = None
    object_type: Optional[str] = None
    main_material: Optional[str] = None
    component_material_ii: Optional[str] = None
    component_material_iii: Optional[str] = None
    title: Optional[str] = None
    title2: Optional[str] = None
    artists_nationality: Optional[str] = None
    country: Optional[str] = None
    origin_place: Optional[str] = None
    find_place: Optional[str] = None
    culture: Optional[str] = None
    dimensions: Optional[str] = None
    weight: Optional[str] = None
    main_artist: Optional[str] = None
    artist_life_date: Optional[str] = None
    author: Optional[str] = None
    scribe: Optional[str] = None
    period_year: Optional[str] = None
    patron_dynasty: Optional[str] = None
    historical_note: Optional[str] = None
    provenance: Optional[str] = None
    brief_description: Optional[str] = None
    detailed_description: Optional[str] = None
    manufacturing_technique: Optional[str] = None
    style: Optional[str] = None
    school: Optional[str] = None
    inscription: Optional[str] = None
    tribe: Optional[str] = None
    costume: Optional[str] = None
    medium: Optional[str] = None
    subject: Optional[str] = None
    language: Optional[str] = None
    script: Optional[str] = None
    number_of_folios: Optional[str] = None
    number_of_illustrations: Optional[str] = None
    coin_description_obverse: Optional[str] = None
    coin_description_reverse: Optional[str] = None
    denomination: Optional[str] = None
    mint: Optional[str] = None

@app.function(
    image=image,
    volumes={"/outputs": modal.Volume.from_name("museum-outputs", create_if_missing=True)},
    gpu="T4",
    timeout=18000,  # 5 hours
    memory=16384,   # 16GB memory
)
def run_museum_pipeline():
    """Main function that combines data and generates embeddings"""
    import clip
    from PIL import Image, UnidentifiedImageError
    
    # Step 1: Combine all JSON files into one
    print("Step 1: Combining JSON files...")
    
    data_files = [
        "ngma_blr.json",
        "nkm_hyd.json", 
        "nat_del.json",
        "im_kol.json",
        "gom_goa.json",
        "alh_ald.json"
    ]
    
    def convert_museum_record(record: Dict[str, Any]) -> Dict[str, Any]:
        """Convert a record from the input format to MuseumRecord format"""
        info = record.get("info", {})
        
        converted = {
            "accession_number": info.get("Accession Number", "unknown_" + str(time.time())),
            "museum_name": info.get("Museum Name", "Unknown Museum"),
            "image_url": record.get("image_url", ""),
            "gallery_name": info.get("Gallery Name"),
            "object_type": info.get("Object Type"),
            "main_material": info.get("Main Material"),
            "title": info.get("Title"),
            "artists_nationality": info.get("Artist's Nationality"),
            "main_artist": info.get("Main Artist"),
            "artist_life_date": info.get("Artist's Life Date / Bio Data"),
            "dimensions": info.get("Dimensions"),
            "medium": info.get("Medium"),
            "brief_description": info.get("Brief Description")
        }
        
        # Add other fields as None
        for field in [
            "component_material_ii", "component_material_iii", "title2",
            "country", "origin_place", "find_place", "culture", "weight",
            "author", "scribe", "period_year", "patron_dynasty", "historical_note",
            "provenance", "detailed_description", "manufacturing_technique", "style",
            "school", "inscription", "tribe", "costume", "subject", "language",
            "script", "number_of_folios", "number_of_illustrations",
            "coin_description_obverse", "coin_description_reverse", "denomination", "mint"
        ]:
            converted[field] = None
        
        return converted
    
    # Load and combine all records
    all_records = []
    
    for file_name in data_files:
        file_path = f"/{file_name}"  # Fixed: files are in root directory
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                print(f"Processing {file_name}...")
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            record_json = json.loads(line)
                            converted_record = convert_museum_record(record_json)
                            museum_record = MuseumRecord(**converted_record)
                            all_records.append(museum_record)
                        except Exception as e:
                            print(f"Error processing record: {e}")
        except FileNotFoundError:
            print(f"File {file_name} not found at {file_path}")
    
    print(f"Loaded {len(all_records)} total records")
    
    # Save combined data as JSONL
    combined_file = "/outputs/all_data.jsonl"
    with open(combined_file, 'w', encoding='utf-8') as outfile:
        for record in all_records:
            # Convert back to original format for saving
            record_dict = {
                "image_url": record.image_url,
                "info": {
                    "Accession Number": record.accession_number,
                    "Museum Name": record.museum_name,
                    "Gallery Name": record.gallery_name,
                    "Object Type": record.object_type,
                    "Main Material": record.main_material,
                    "Title": record.title,
                    "Artist's Nationality": record.artists_nationality,
                    "Main Artist": record.main_artist,
                    "Artist's Life Date / Bio Data": record.artist_life_date,
                    "Dimensions": record.dimensions,
                    "Medium": record.medium,
                    "Brief Description": record.brief_description
                }
            }
            json.dump(record_dict, outfile, ensure_ascii=False)
            outfile.write('\n')
    
    print(f"Combined data saved to {combined_file}")
    
    # Step 2: Load CLIP model
    print("Step 2: Loading CLIP model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f"CLIP model loaded on {device}")
    
    # Step 3: Generate embeddings
    print("Step 3: Generating embeddings...")
    
    def get_image(url: str) -> Optional[requests.Response]:
        req_count = 0
        while req_count < 3:
            try:
                response = requests.get(url, timeout=10, verify=False,
                                      headers={'User-Agent': 'Mozilla/5.0 (compatible; MuseumBot/1.0)'})
                if response.status_code not in (200, 404):
                    print(f"Retrying {url[:50]}..., status code: {response.status_code}")
                    req_count += 1
                    time.sleep(1)
                else:
                    return response
            except requests.exceptions.RequestException as e:
                print(f"Retrying {url[:50]}..., error: {e}")
                req_count += 1
                time.sleep(3)
        print(f"Failed to get image from {url[:50]}... after 3 retries")
        return None
    
    def create_image_embeddings(image_urls: List[str]) -> Tuple[List[int], np.ndarray]:
        """Create image embeddings"""
        with ThreadPoolExecutor(max_workers=10) as executor:
            responses = list(executor.map(get_image, image_urls))
        
        failed_images = []
        success_images = []
        
        for i, response in enumerate(responses):
            if response is None or response.status_code != 200:
                failed_images.append(i)
                continue
            try:
                img = Image.open(BytesIO(response.content))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                success_images.append(img)
            except UnidentifiedImageError:
                print(f"Failed to open image {i}")
                failed_images.append(i)
        
        if not success_images:
            return failed_images, np.array([])
        
        print(f"Processing {len(success_images)} images...")
        images = [preprocess(img) for img in success_images]
        image_tensors = torch.stack(images).to(device)
        
        with torch.no_grad():
            embeddings = model.encode_image(image_tensors).cpu().numpy()
        
        return failed_images, embeddings
    
    # Extract image URLs
    image_urls = [record.image_url for record in all_records]
    
    # Generate embeddings in batches
    batch_size = 50
    all_embeddings = []
    failed_indices = []
    
    for i in range(0, len(image_urls), batch_size):
        print(f"Processing batch {i//batch_size + 1}/{(len(image_urls) + batch_size - 1)//batch_size}")
        batch_urls = image_urls[i:i+batch_size]
        batch_failed, batch_embeddings = create_image_embeddings(batch_urls)
        
        # Adjust failed indices to global indices
        batch_failed = [idx + i for idx in batch_failed]
        failed_indices.extend(batch_failed)
        
        if len(batch_embeddings) > 0:
            all_embeddings.append(batch_embeddings)
    
    if not all_embeddings:
        print("No embeddings were generated. Check image URLs.")
        return {"error": "No embeddings generated"}
    
    # Combine all embeddings
    combined_embeddings = np.vstack(all_embeddings)
    
    # Create list of successful record IDs
    successful_indices = [i for i in range(len(all_records)) if i not in failed_indices]
    successful_ids = np.array([all_records[i].accession_number.encode('utf-8') for i in successful_indices])
    
    # Save embeddings to HDF5 file
    output_file = "/outputs/embeddings.h5"
    with h5py.File(output_file, "w") as f:
        f.create_dataset("embeddings", data=combined_embeddings)
        f.create_dataset("ids", data=successful_ids)
        f.create_dataset("record_indices", data=np.array(successful_indices))
        f.create_dataset("metadata", data=json.dumps({
            "model": "CLIP ViT-B/32",
            "total_records": len(all_records),
            "successful_records": len(successful_ids),
            "failed_records": len(failed_indices),
            "batch_size": batch_size
        }).encode('utf-8'))
    
    print(f"Saved {len(successful_ids)} embeddings to {output_file}")
    print(f"Failed to process {len(failed_indices)} images")
    
    return {
        "total_records": len(all_records),
        "successful_embeddings": len(successful_ids),
        "failed_images": len(failed_indices),
        "output_file": output_file
    }

@app.function(
    image=image,
    volumes={"/outputs": modal.Volume.from_name("museum-outputs", create_if_missing=True)},
    timeout=600,
)
def list_output_files():
    """List all files in the output directory"""
    import os
    output_dir = "/outputs"
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        print(f"Files in {output_dir}:")
        for file in files:
            file_path = os.path.join(output_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  {file} ({size:,} bytes)")
        return files
    else:
        print(f"Directory {output_dir} does not exist")
        return []

# Local entrypoints
@app.local_entrypoint()
def main():
    """Main entrypoint - runs the complete pipeline"""
    print("Starting museum embedding pipeline...")
    results = run_museum_pipeline.remote()
    print("\nPipeline Results:")
    for key, value in results.items():
        print(f"{key}: {value}")

@app.local_entrypoint()
def list_files():
    """List output files"""
    files = list_output_files.remote()
    return files