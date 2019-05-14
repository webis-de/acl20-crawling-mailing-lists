#!/usr/bin/env python3

import json
import sys

labels_dict = {
    120: "raw_code",
    121: "technical",
    122: "log_data",
    123: "closing",
    124: "mua_signature",
    125: "patch",
    126: "salutation",
    127: "personal_signature",
    128: "paragraph",
    129: "quotation_marker",
    130: "tabular",
    131: "quotation",
    132: "visual_separator",
    133: "inline_headers",
    134: "section_heading"
}

for document in sys.stdin:
    json_doc = json.loads(document)

    converted_doc = json_doc.copy()

    labels = []
    annotations = json_doc["annotations"]

    for annotation in annotations:
        annotation["label"] = labels_dict[annotation["label"]]
        start_offset = annotation["start_offset"]
        end_offset = annotation["end_offset"]
        label = annotation["label"]

        labels.append((start_offset, end_offset, label))

    converted_doc["labels"] = labels
    print(json.dumps(converted_doc))
