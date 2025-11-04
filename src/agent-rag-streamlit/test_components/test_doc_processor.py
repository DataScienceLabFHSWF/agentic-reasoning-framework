import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils.doc_processor import DocProcessor

data_path = "../data"
processed_path = "../processed_files"

processor = DocProcessor()
docs = processor.process_directory(data_path, processed_path)

print(len(docs), "documents processed.")




