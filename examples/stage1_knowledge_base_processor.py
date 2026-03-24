import os

from dotenv import load_dotenv

from agentrf.doc_processing import DocProcessor
from agentrf.settings import load_settings

# 1. Load the config dynamically
load_dotenv()
config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

# 2. Extract paths for the knowledge base
kb_path = settings.paths.knowledge_base_raw
output_path = settings.paths.knowledge_base_processed

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
