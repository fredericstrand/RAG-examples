from pathlib import Path
from docling.document_converter import DocumentConverter
from collections import defaultdict
import json
import re

converter = DocumentConverter()
data_dir = Path("chunk-methods/data/")

document_types = {
    "lgsiv": "Lagmannsrettens avgjørelse",
    "hgr":   "Høyesteretts avgjørelse",
    "nl":    "Lovtekst",
    "sf":    "Sentrale Forskrifter",
}

all_out = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"title": None, "text": ""})))

for doc in sorted(data_dir.glob("*.pdf")):
    result = converter.convert(str(doc))
    doc_data = result.document.export_to_dict()

    doc_name = doc_data.get("name", doc.stem)
    m = re.match(r"^([^_]+)", doc_name)
    prefix = m.group(1) if m else ""
    doc_type = document_types.get(prefix, "Ukjent dokumenttype")

    for item in doc_data.get("texts", []):
        prov = item.get("prov") or []
        if not prov:
            print("no provenance for item", item)
            continue
        page_num = prov[0].get("page_no")
        if page_num is None:
            print("no page number for item", item)
            continue

        label = item.get("label")
        text = (item.get("text") or "").strip()
        if not text:
            print("no text for item", item)
            continue

        output = all_out[doc_type][doc_name][page_num]
        if label == "section_header":
            output["title"] = text
        else:
            if output["text"]:
                output["text"] += " "
            output["text"] += text

def to_dict(x):
    if isinstance(x, dict):
        return {k: to_dict(v) for k, v in x.items()}
    return x

output_path = Path("parsed_output.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(to_dict(all_out), f, ensure_ascii=False, indent=2)
