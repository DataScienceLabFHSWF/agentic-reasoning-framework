import os

from dotenv import load_dotenv

from agentrf.doc_processing import DocProcessor
from agentrf.settings import load_settings

# 1. Load the config dynamically
load_dotenv()
config_path = os.getenv("AGENTRF_CONFIG")
settings = load_settings(config_path)

# 2. Extract paths
kb_path = settings.paths.knowledge_base_raw
output_path = settings.paths.uploads_tmp

# 3. Pass paths to the processor
processor = DocProcessor()

doc = processor.process_document(
    file_path=kb_path / "uvu.pdf",
    output_dir=output_path,
)

if doc:
    print(f"Successfully processed to: {doc.processed_path}")
