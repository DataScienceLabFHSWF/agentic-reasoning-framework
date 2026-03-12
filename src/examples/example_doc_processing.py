from pathlib import Path
from config.loader import Config
from agentrf.doc_processing import DocProcessor

config = Config(Path(__file__).resolve().parents[1] / "config" / "toy.yaml")
processor = DocProcessor()

# Specify a single document file, not a directory
file_path = Path(config.path("knowledge_base_raw")) / "uvu.pdf"

doc = processor.process_document(
    file_path=file_path,
    output_dir=config.path("uploads_tmp"),
)

if doc:
    print(f"Source : {doc.source_path}")
    print(f"Saved  : {doc.processed_path}")
    print(f"Preview:\n{doc.text[:500]}")
else:
    print("Processing failed.")
