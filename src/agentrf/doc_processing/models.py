from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ProcessedDocument:
    text: str
    source_path: str
    file_type: str
    processed_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
