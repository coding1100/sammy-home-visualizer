from typing import List, Dict, Any
from xml.etree.ElementTree import Element, SubElement, tostring
from dataclasses import dataclass
import base64

@dataclass
class SVGItem:
    id: str
    category: str
    address: str
    points: str
    score: float

SVG_HEADER_STYLE = """
  .interactive { pointer-events: all; cursor: pointer; }
  [data-selected="true"] { outline: 2px dashed #4AA3FF; }
"""


def build_svg(width: int, height: int, items: List[SVGItem], origin_name: str) -> str:
    svg = Element('svg', attrib={
        'xmlns': 'http://www.w3.org/2000/svg',
        'viewBox': f"0 0 {width} {height}",
    })
    defs = SubElement(svg, 'defs')
    style = SubElement(defs, 'style')
    style.text = SVG_HEADER_STYLE

    house = SubElement(svg, 'g', attrib={'id': 'house', 'data-type': 'house', 'data-origin': origin_name})

    groups: Dict[str, Element] = {}
    for it in items:
        grp = groups.get(it.category)
        if grp is None:
            grp = SubElement(svg, 'g', attrib={'id': it.category.replace(' ', '_'), 'class': 'interactive', 'data-type': it.category})
            groups[it.category] = grp
        SubElement(grp, 'polygon', attrib={
            'id': it.id,
            'points': it.points,
            'data-type': it.category,
            'data-address': it.address,
            'data-confidence': f"{it.score:.2f}",
        })

    return tostring(svg, encoding='unicode')