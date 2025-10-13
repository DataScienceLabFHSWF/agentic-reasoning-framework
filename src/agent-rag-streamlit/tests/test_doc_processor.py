import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils.doc_processor import DocProcessor

processor = DocProcessor()
docs = processor.process_directory("/home/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/raw_data", "/home/rrao/projects/agentic-reasoning-framework/src/agent-rag-streamlit/processed_files")

print(len(docs), "documents processed.")




