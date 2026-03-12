from pathlib import Path
from config.loader import Config
from agentrf.doc_processing import DocProcessor

config = Config(Path(__file__).resolve().parents[1] / "config" / "toy.yaml")
processor = DocProcessor()

docs = processor.process_knowledge_base(
    input_dir=config.path("knowledge_base_raw"),
    output_dir=config.path("knowledge_base_processed"),
    save_output=config.get("processing", "save_markdown"),
)

print(f"Processed {len(docs)} documents.")
for d in docs[:5]:
    print(f"- {d.source_path} -> {d.processed_path}")
