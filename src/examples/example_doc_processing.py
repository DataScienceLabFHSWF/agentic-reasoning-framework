from agentrf.config import load_config
from agentrf.doc_processing import DocProcessor

config = load_config()

processor = DocProcessor()

doc = processor.process_document(
    file_path=config.path("knowledge_base_raw") / "KKG_Stilllegung.pdf",
    output_dir=config.path("uploads_tmp"),
)

if doc:
    print(f"Source : {doc.source_path}")
    print(f"Saved  : {doc.processed_path}")
    print(f"Preview:\n{doc.text[:500]}")
else:
    print("Processing failed.")
