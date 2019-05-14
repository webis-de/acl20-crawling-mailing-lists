#!/usr/bin/env python3

import json
import sys

# Gold 700 (1)
# labels_dict = {
#     21: "raw_code",
#     24: "technical",
#     27: "log_data",
#     15: "closing",
#     17: "mua_signature",
#     18: "patch",
#     22: "salutation",
#     19: "personal_signature",
#     25: "paragraph",
#     16: "quotation_marker",
#     26: "tabular",
#     20: "quotation",
#     28: "visual_separator",
#     29: "inline_headers",
#     30: "section_heading"
# }


# Gold 300 (2)
# labels_dict = {
#     59: "raw_code",
#     65: "technical",
#     66: "log_data",
#     57: "closing",
#     58: "mua_signature",
#     53: "patch",
#     52: "salutation",
#     63: "personal_signature",
#     61: "paragraph",
#     60: "quotation_marker",
#     62: "tabular",
#     56: "quotation",
#     64: "visual_separator",
#     55: "inline_headers",
#     54: "section_heading"
# }

# Gold 1k (3)
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
