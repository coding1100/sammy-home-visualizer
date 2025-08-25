from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class ElementJSON:
    id: str
    address: str
    category: str
    polygon: list  # [[x,y], ...]
    bbox: list     # [x,y,w,h]
    confidence: float
    attributes: Dict[str, Any]

@dataclass
class AnalysisJSON:
    image: Dict[str, Any]
    elements: List[ElementJSON]
    relations: List[Dict[str, str]]
    version: str = "0.1.0"

    def to_dict(self):
        return {
            'image': self.image,
            'elements': [asdict(e) for e in self.elements],
            'relations': self.relations,
            'version': self.version,
        }