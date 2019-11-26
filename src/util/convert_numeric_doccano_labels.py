#!/usr/bin/env python3

# Converter for numeric Doccano labels to text labels. Needed for re-import into Doccano.
# Adjust the numeric labels in-code according to your Doccano database.

import json

import click


labels_dict = {
    357: "paragraph",
    358: "tabular",
    359: "raw_code",
    360: "section_heading",
    361: "visual_separator",
    362: "inline_headers",
    363: "patch",
    364: "technical",
    365: "log_data",
    366: "closing",
    367: "quotation_marker",
    368: "personal_signature",
    369: "quotation",
    370: "salutation",
    371: "mua_signature"
}


@click.command()
@click.argument('files', nargs=-1)
def main(files):
    for file in files:
        json_doc = json.loads(file)

        labels = []
        annotations = list(json_doc["annotations"])

        converted_doc = dict(json_doc)
        del converted_doc['annotations']

        for annotation in annotations:
            annotation["label"] = labels_dict[annotation["label"]]
            start_offset = annotation["start_offset"]
            end_offset = annotation["end_offset"]
            label = annotation["label"]

            labels.append((start_offset, end_offset, label))

        converted_doc["labels"] = labels
        click.echo(json.dumps(converted_doc))


if __name__ == '__main__':
    main()
