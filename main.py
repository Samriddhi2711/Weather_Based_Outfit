"""
Weather-aware Outfit Recommender (FAISS + FastAPI)
--------------------------------------------------
Endpoints:
 - POST /recommend : Takes event, style, location, preferences → returns recommended outfits
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import os
import requests
import openai
import json
import faiss
import numpy as np
import dotenv

# ---------- Load env vars ----------
dotenv.load_dotenv()  
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY")  
PRODUCT_EMBEDDINGS_FILE = "products_with_embeddings.json"

if not OPENAI_API_KEY:
    raise RuntimeError("Please set OPENAI_API_KEY in .env")

openai.api_key = OPENAI_API_KEY

# ---------- FastAPI app ----------
app = FastAPI(title="Weather-aware Outfit Recommender (FAISS)")

# ---------- Pydantic models ----------
class RecommendRequest(BaseModel):
    event: str
    style: Optional[str] = None
    location: Optional[str] = None  
    preferences: Optional[str] = None
    k: Optional[int] = 6

class ProductItem(BaseModel):
    id: str
    title: str
    description: str
    price: float
    tags: List[str]
    image_url: Optional[str]

class RecommendResponse(BaseModel):
    weather: Dict[str, Any]
    query_vector: Optional[List[float]]
    results: List[ProductItem]
    explanation: Dict[str, Any]  # allow JSON object

# ---------- Load FAISS index ----------
print("Loading FAISS index and product embeddings...")
with open(PRODUCT_EMBEDDINGS_FILE, "r") as f:
    products_with_emb = json.load(f)

vectors = np.array([p["vector"] for p in products_with_emb]).astype("float32")
metadata = [p["item"] for p in products_with_emb]

dim = vectors.shape[1]
faiss_index = faiss.IndexFlatL2(dim)
faiss_index.add(vectors)
print(f"FAISS index loaded with {faiss_index.ntotal} vectors")

# ---------- Helpers ----------
def query_faiss(qvec, top_k=5):
    qvec_np = np.array([qvec]).astype("float32")
    distances, indices = faiss_index.search(qvec_np, top_k)
    results = [metadata[i] for i in indices[0]]
    return results

def fetch_weather_for_location(location: Optional[str]):
    if not WEATHER_API_KEY:
        return {"error": "No weather API key configured"}
    if not location:
        return {"error": "No location provided"}

    url = f"https://api.openweathermap.org/data/2.5/weather?q={location}&units=metric&appid={WEATHER_API_KEY}"
    try:
        r = requests.get(url, timeout=6.0)
        r.raise_for_status()
        data = r.json()
        weather = {
            "description": data["weather"][0]["description"],
            "main": data["weather"][0]["main"],
            "temp_c": data["main"]["temp"],
            "feels_like_c": data["main"]["feels_like"],
            "wind_m_s": data["wind"]["speed"],
        }
        return weather
    except Exception as e:
        return {"error": str(e)}

def build_semantic_query(event: str, style: Optional[str], weather: Dict[str,Any], preferences: Optional[str]):
    parts = [f"Event: {event}"]
    if style:
        parts.append(f"Style: {style}")
    if weather and "temp_c" in weather:
        parts.append(f"Weather: {weather['main']}, {weather['temp_c']}°C, feels like {weather.get('feels_like_c')}°C")
    if preferences:
        parts.append(f"Preferences: {preferences}")
    query_text = " | ".join(parts)

    emb_resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )
    qvec = emb_resp.data[0].embedding
    return query_text, qvec

def ask_llm_to_format(event: str, style: Optional[str], weather: Dict[str,Any], preferences: Optional[str], items: List[Dict]):
    system = (
        "You are a helpful stylist assistant. Given an event, style, weather, and a list of product items, "
        "produce a JSON object with 'title', 'summary', and 'looks'. Each look should contain: 'name', 'items' (list of product ids), "
        "'quick_reason' (1 sentence), and 'detailed_reason' (2-3 sentences). Keep language concise and human-friendly."
    )

    weather_str = f"{weather.get('main','')}, {weather.get('temp_c','?')}°C"

    user_msg = {
        "event": event,
        "style": style,
        "weather": weather_str,
        "preferences": preferences,
        "catalog": items,
        "instructions": "Create 2-4 curated looks (combinations of items). Use available items; do not invent product ids."
    }

    prompt = [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user_msg)}
    ]

    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=0.7,
        max_tokens=600
    )

    out_text = resp.choices[0].message.content
    try:
        parsed = json.loads(out_text)
        return parsed
    except Exception:
        return {"raw": out_text}

# ---------- Main Endpoint ----------
@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    weather = fetch_weather_for_location(req.location) if req.location else {"error": "no location"}
    query_text, qvec = build_semantic_query(req.event, req.style, weather, req.preferences)

    try:
        raw_items = query_faiss(qvec, top_k=req.k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"FAISS query failed: {e}")

    items_for_llm = [{"id": it["id"], "title": it["title"], "description": it["description"],
                      "price": it["price"], "tags": it["tags"], "image_url": it.get("image_url")} for it in raw_items]

    try:
        explanation = ask_llm_to_format(req.event, req.style, weather, req.preferences, items_for_llm)
    except Exception as e:
        explanation = {"error": f"LLM formatting failed: {e}"}

    results = [ProductItem(**it) for it in items_for_llm]

    # ✅ Return proper JSON (not stringified)
    return {
        "weather": weather,
        "query_vector": qvec[:8],  # truncated for readability
        "results": results,
        "explanation": explanation   # dict as JSON
    }
