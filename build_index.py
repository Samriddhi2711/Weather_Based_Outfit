import json
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()  # .env file me OPENAI_API_KEY hona chahiye

# Step 1: Load catalog
with open("product_catalog.json", "r") as f:
    products = json.load(f)
    print("Products loaded:", len(products))
    if len(products) > 0:
        print("First product sample:", products[0])

# Step 2: Prepare text for embeddings
texts = [f"{p['title']} - {p['description']} Tags: {', '.join(p['tags'])}" for p in products]
print("Texts prepared:", len(texts))

# Step 3: OpenAI embeddings (LangChain wrapper)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Step 4: Create FAISS index with metadata
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=products)

# Step 5: Save FAISS index locally
os.makedirs("index", exist_ok=True)
vectorstore.save_local("index")
print("✅ FAISS index build ho gaya aur `index/` folder me save ho gaya!")

# ------------------------------
# Step 6: Generate products_with_embeddings.json
# ------------------------------
print("🔹 Creating `products_with_embeddings.json` with embeddings...")

# embeddings.embed_documents returns a list of vectors
vectors = embeddings.embed_documents(texts)
print("Vectors generated:", len(vectors))

products_with_embeddings = []
for p, vec in zip(products, vectors):
    print("Adding:", p["id"])
    products_with_embeddings.append({
        "id": p["id"],
        "item": p,
        "vector": vec
    })

if products_with_embeddings:
    with open("products_with_embeddings.json", "w") as f:
        json.dump(products_with_embeddings, f, indent=2)
    print("✅ `products_with_embeddings.json` ban gaya with data! Size:", len(products_with_embeddings))
else:
    print("❌ products_with_embeddings empty hai!")
