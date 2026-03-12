import yaml
from pathlib import Path

# Import the pure code from your framework
from agentrf.doc_processing import DocProcessor

# 1. Load the config strictly in the consumer application using an absolute path
config_file = "/home/rrao/projects/agents/agentic-reasoning-framework/src/config/toy.yaml"

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# 2. Extract paths (assuming toy.yaml provides absolute paths)
kb_path = Path(config["paths"]["knowledge_base_raw"])
output_path = Path(config["paths"]["uploads_tmp"])

# 3. Pass raw paths to agentrf
processor = DocProcessor()

doc = processor.process_document(
    file_path=kb_path / "uvu.pdf",
    output_dir=output_path,
)

if doc:
    print(f"Successfully processed to: {doc.processed_path}")