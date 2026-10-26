#!/usr/bin/env python3
"""Validate image/label joins and report the training cohort without pandas."""

import argparse
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from pathlib import Path


XML_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--label-file", type=Path, default=Path("data/labels.xlsx"))
    parser.add_argument("--r-threshold", type=float, default=0.75)
    return parser.parse_args()


def column_index(cell_reference):
    """Convert an Excel column reference such as AA12 to a zero-based index."""
    letters = re.match(r"[A-Z]+", cell_reference).group(0)
    index = 0
    for letter in letters:
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def read_first_sheet(path):
    """Read primitive values from the first XLSX worksheet using the stdlib."""
    with zipfile.ZipFile(path) as workbook:
        shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.findall(".//x:t", XML_NS))
            for item in shared_root.findall("x:si", XML_NS)
        ]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    rows = []
    for row in sheet.findall(".//x:sheetData/x:row", XML_NS):
        values = {}
        for cell in row.findall("x:c", XML_NS):
            value_node = cell.find("x:v", XML_NS)
            if value_node is None:
                continue
            value = value_node.text
            if cell.get("t") == "s":
                value = shared[int(value)]
            else:
                value = float(value)
            values[column_index(cell.get("r"))] = value
        if values:
            rows.append([values.get(index) for index in range(max(values) + 1)])
    return rows


def validate(image_dir, label_file, r_threshold):
    """Check required fields and summarize records that pass training filters."""
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_file.is_file():
        raise FileNotFoundError(f"Label file not found: {label_file}")

    table = read_first_sheet(label_file)
    headers = table[0]
    required = {"filename", "modulus", "R", "day"}
    if not required.issubset(headers):
        raise ValueError(f"Missing columns: {sorted(required.difference(headers))}")

    records = []
    for values in table[1:]:
        padded = values + [None] * (len(headers) - len(values))
        records.append(dict(zip(headers, padded)))

    image_names = {
        path.name
        for pattern in ("*.tif", "*.tiff")
        for path in image_dir.glob(pattern)
    }
    allowed_days = {1, 4, 7, 10, 13}
    valid = []
    for record in records:
        try:
            modulus = float(record["modulus"])
            r_value = float(record["R"])
            day = int(record["day"])
        except (TypeError, ValueError):
            continue
        if (
            record["filename"] in image_names
            and math.isfinite(modulus)
            and modulus > 0
            and math.isfinite(r_value)
            and r_value > r_threshold
            and day in allowed_days
        ):
            valid.append(day)

    if not valid:
        raise ValueError("No image/label pair passed the training filters.")
    day_counts = Counter(valid)
    group_counts = {
        "Day_1": day_counts[1],
        "Day_4_7": day_counts[4] + day_counts[7],
        "Day_10_13": day_counts[10] + day_counts[13],
    }
    print(f"Workbook records: {len(records)}")
    print(f"Extracted images: {len(image_names)}")
    print(f"Valid training pairs (R > {r_threshold}): {len(valid)}")
    print(f"Counts by day: {dict(sorted(day_counts.items()))}")
    print(f"Counts by model class: {group_counts}")


if __name__ == "__main__":
    args = parse_args()
    validate(args.image_dir, args.label_file, args.r_threshold)
