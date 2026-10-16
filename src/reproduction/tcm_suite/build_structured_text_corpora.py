#!/usr/bin/env python3
"""Build deterministic, auditable TCM-suite text-control corpora.

The script never calls an LLM.  It converts mapping-table fields into fixed
English templates and writes one JSONL record per model entity, preserving the
mapping CSV row order and the explicit entity identifier.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def clean(value):
    if value is None or pd.isna(value):
        return None
    value = " ".join(str(value).split())
    return value if value and value.lower() != "nan" else None


def first(row, columns):
    for column in columns:
        value = clean(row.get(column))
        if value:
            return value
    return None


def add(parts, label, value):
    value = clean(value)
    if value:
        parts.append(f"{label}: {value}.")


def herb_texts(row):
    entity_id = str(row["herb_id"])
    canonical = first(row, ["englishName", "latinName", "pinyinName", "xbChineseName"])
    canonical = canonical or f"herb entity {entity_id}"

    name_parts = [f"Herbal entity name: {canonical}."]
    for label, column in [
        ("Pinyin name", "pinyinName"),
        ("Latin or botanical name", "latinName"),
        ("Chinese name", "xbChineseName"),
    ]:
        value = clean(row.get(column))
        if value and value != canonical:
            add(name_parts, label, value)

    clean_parts = list(name_parts)
    add(clean_parts, "Intrinsic properties", row.get("properties"))
    add(clean_parts, "Meridian attribution", row.get("meridians"))
    add(clean_parts, "Medicinal part or material", row.get("usePart"))

    indication_parts = list(clean_parts)
    add(indication_parts, "Recorded effect", row.get("effect"))
    add(indication_parts, "Recorded indication or pharmacological description", row.get("indication"))

    return canonical, {
        "name_only": " ".join(name_parts),
        "structured_clean": " ".join(clean_parts),
        "structured_with_indication": " ".join(indication_parts),
    }


def disease_texts(row):
    entity_id = str(row["disease_id"])
    canonical = first(row, ["englishName", "xbDiseaseName"])
    canonical = canonical or f"biomedical entity {entity_id}"

    name_parts = [f"Biomedical entity name: {canonical}."]
    alternative = clean(row.get("xbDiseaseName"))
    if alternative and alternative != canonical:
        add(name_parts, "Alternative disease name", alternative)

    clean_parts = list(name_parts)
    add(clean_parts, "ICD-9 code", row.get("xbIcd9"))
    add(clean_parts, "ICD-10 code", row.get("xbIcd10"))

    # The disease mapping has no herb/compound association field.  Therefore
    # the clean and with-indication disease templates are intentionally equal;
    # only the herb-side condition changes when effect/indication is enabled.
    return canonical, {
        "name_only": " ".join(name_parts),
        "structured_clean": " ".join(clean_parts),
        "structured_with_indication": " ".join(clean_parts),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_corpora(entity, mapping_path, output_dir):
    id_column = "herb_id" if entity == "herb" else "disease_id"
    builder = herb_texts if entity == "herb" else disease_texts
    frame = pd.read_csv(mapping_path, encoding="utf-8-sig", dtype=str)
    if frame[id_column].duplicated().any():
        duplicates = frame.loc[frame[id_column].duplicated(), id_column].tolist()
        raise ValueError(f"duplicate {id_column} values: {duplicates[:10]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        variant: output_dir / f"{variant}_{entity}.jsonl"
        for variant in ("name_only", "structured_clean", "structured_with_indication")
    }
    streams = {variant: path.open("w", encoding="utf-8") for variant, path in paths.items()}
    try:
        for row_index, row in frame.iterrows():
            canonical, texts = builder(row)
            for variant, description in texts.items():
                record = {
                    "entity": entity,
                    "variant": variant,
                    "row_index": int(row_index),
                    "entity_id": str(row[id_column]),
                    "entity_name": canonical,
                    "description": description,
                    "source": "deterministic_mapping_template",
                }
                streams[variant].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for stream in streams.values():
            stream.close()

    return {
        variant: {"path": str(path.resolve()), "rows": len(frame), "sha256": sha256(path)}
        for variant, path in paths.items()
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--herb-mapping", type=Path, required=True)
    parser.add_argument("--disease-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = {
        "herb_mapping": str(args.herb_mapping.resolve()),
        "disease_mapping": str(args.disease_mapping.resolve()),
        "herb_mapping_sha256": sha256(args.herb_mapping),
        "disease_mapping_sha256": sha256(args.disease_mapping),
        "definition": {
            "name_only": "entity names only",
            "structured_clean": "intrinsic mapping fields; herb effect and indication excluded",
            "structured_with_indication": "structured_clean plus herb effect and indication",
            "disease_side_note": "disease mapping has no herb field, so its clean and with-indication texts are identical",
        },
        "outputs": {},
    }
    manifest["outputs"]["herb"] = write_corpora("herb", args.herb_mapping, args.output_dir)
    manifest["outputs"]["disease"] = write_corpora("disease", args.disease_mapping, args.output_dir)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
