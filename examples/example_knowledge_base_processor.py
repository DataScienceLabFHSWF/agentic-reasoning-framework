import yaml
from pathlib import Path

# Import the pure code from your framework
from agentrf.doc_processing import DocProcessor

# 1. Load the config strictly in the consumer application using an absolute path
config_file = "/home/rrao/projects/agents/agentic-reasoning-framework/src/config/toy.yaml"

with open(config_file, "r") as f:
    config = yaml.safe_load(f)

# 2. Extract paths for the knowledge base
kb_path = Path(config["paths"]["knowledge_base_raw"])
output_path = Path(config["paths"]["knowledge_base_processed"])

# 3. Initialize processor
processor = DocProcessor()

# 4. Process the entire directory
print(f"Processing knowledge base from: {kb_path}")
processed_docs = processor.process_knowledge_base(
    input_dir=kb_path,
    output_dir=output_path,
)

# 5. Output results
print(f"\nSuccessfully processed {len(processed_docs)} documents:")
for doc in processed_docs:
    print(f" - Saved {doc.metadata['source_name']} -> {doc.processed_path}")