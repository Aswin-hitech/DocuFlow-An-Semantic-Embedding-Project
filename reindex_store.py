import os
import sys

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion.document_parser import DocumentParser
from retrieval.search import SemanticSearch

def reindex_all():
    docs_dir = os.path.join("data", "raw", "documents")
    store_path = os.path.join("data", "embeddings", "vector_store.json")

    print(f"🔄 Initializing SemanticSearch engine...")
    search_engine = SemanticSearch(storage_path=store_path)

    # Clear old outdated chunks
    print(f"🧹 Clearing existing outdated chunks from vector store...")
    search_engine.vector_store.clear()
    search_engine.chunks = []
    search_engine.chunk_embeddings = []

    if not os.path.exists(docs_dir):
        print(f"Directory {docs_dir} does not exist.")
        return

    files = [f for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))]
    print(f"📁 Found {len(files)} files to index: {files}\n")

    for filename in files:
        file_path = os.path.join(docs_dir, filename)
        print(f"📄 Parsing and indexing '{filename}'...")
        try:
            parsed = DocumentParser.parse_file(file_path)
            extracted_text = parsed.get("text", "")
            if not extracted_text.strip():
                print(f"  ⚠️ Skipping {filename}: no readable text.")
                continue

            metadata = parsed.get("metadata", {})
            metadata["original_filename"] = filename

            chunk_ids = search_engine.index_document(
                text=extracted_text,
                doc_id=filename,
                source=filename,
                chunk_size=500,
                overlap=50,
                metadata=metadata
            )
            print(f"  ✅ Indexed {len(chunk_ids)} boundary-aware chunks for '{filename}'.")
        except Exception as e:
            print(f"  ❌ Error indexing {filename}: {e}")

    stats = search_engine.get_stats()
    print(f"\n==========================================")
    print(f"🎉 Reindexing complete!")
    print(f"📊 Total indexed chunks: {stats.get('total_chunks')}")
    print(f"📐 Vector dimension:     {stats.get('dimension')}")
    print(f"💾 Storage file:         {store_path}")
    print(f"==========================================\n")

    # Verify query
    print("🔍 Testing search for 'types of noball':")
    results = search_engine.search("types of noball", top_k=2)
    for i, res in enumerate(results, 1):
        print(f"\n--- Result #{i} (Score: {res['score']:.4f}) ---")
        print(res["chunk"])

if __name__ == "__main__":
    reindex_all()
