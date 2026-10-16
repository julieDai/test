"""Generate auditable ASTRA relationship visualizations without extra packages.

The command reads the four relation files used by ASTRA and produces:

* a database-level relation inventory;
* a filtered herb-centred H--C--P--D subgraph;
* editable SVG, publication-ready PDF, interactive HTML, and JSON provenance.

The case view is deliberately filtered.  Drawing every edge at once would turn
the TCM-suite graph into an unreadable hairball and would not constitute useful
evidence.  The JSON output records the deterministic filtering rule and every
displayed edge.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable, Iterator, Sequence


RELATION_FILES = {
    "H-C": "H_C.txt",
    "C-P": "C_P.txt",
    "P-D": "P_D.txt",
    "H-D": "H_D.txt",
}

ENTITY_COLORS = {
    "herb": ("#4E8F62", "#E7F3E9"),
    "compound": ("#D88932", "#FCEBD7"),
    "protein": ("#3E7FA6", "#E1EFF7"),
    "disease": ("#8063A6", "#EEE8F6"),
}

RELATION_COLORS = {
    "H-C": "#D99A4A",
    "C-P": "#4B9A91",
    "P-D": "#8A70AF",
    "H-D": "#D95D5D",
}


@dataclass(frozen=True)
class Node:
    key: str
    entity_type: str
    entity_id: str
    label: str


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str


@dataclass
class CaseGraph:
    herb_ids: list[str]
    disease_ids: list[str]
    nodes: list[Node]
    edges: list[Edge]
    available_counts: dict[str, object]
    selection_note: str


def positive_rows(path: Path) -> Iterator[tuple[str, str]]:
    """Yield positive entity pairs from a whitespace- or comma-delimited file."""

    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [field for field in re.split(r"[\s,]+", line) if field]
            if len(fields) < 2:
                raise ValueError(f"{path}:{line_number}: expected at least two columns")
            if len(fields) >= 3:
                try:
                    if float(fields[2]) <= 0:
                        continue
                except ValueError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid relation label {fields[2]!r}"
                    ) from error
            yield fields[0], fields[1]


def require_relation_files(dataset_dir: Path) -> dict[str, Path]:
    paths = {name: dataset_dir / filename for name, filename in RELATION_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing relation file(s): " + ", ".join(missing))
    return paths


def database_inventory(paths: dict[str, Path]) -> dict[str, object]:
    entity_sets = {name: set() for name in ENTITY_COLORS}
    relation_counts: dict[str, int] = {}
    endpoints = {
        "H-C": ("herb", "compound"),
        "C-P": ("compound", "protein"),
        "P-D": ("protein", "disease"),
        "H-D": ("herb", "disease"),
    }
    for relation, path in paths.items():
        left_type, right_type = endpoints[relation]
        count = 0
        for left, right in positive_rows(path):
            entity_sets[left_type].add(left)
            entity_sets[right_type].add(right)
            count += 1
        relation_counts[relation] = count
    return {
        "entities": {name: len(values) for name, values in entity_sets.items()},
        "relations": relation_counts,
    }


def _mapping_label(entity_type: str, row: dict[str, str]) -> str | None:
    priorities = {
        "herb": ("englishName", "latinName", "pinyinName", "herb_curie", "name"),
        # The latter columns are the field names used by the original
        # TCM-suite v4 entity tables (compound.csv and target_protein.csv).
        "compound": (
            "compound_name", "name", "latinName", "omimName", "pubchem_id", "cid"
        ),
        "protein": (
            "gene_symbol", "gene", "protein_name", "name", "xbTargetName",
            "ensembl_id", "stringcode",
        ),
        "disease": ("englishName", "disease_name", "name", "disease_mondo"),
    }
    for column in priorities[entity_type]:
        value = (row.get(column) or "").strip()
        if not value:
            continue
        if column in {"pubchem_id", "cid"} and value.isdigit():
            return f"PubChem CID {value}"
        if column == "herb_curie" and value.lower().startswith("ncbitaxon:"):
            return "NCBI Taxon " + value.split(":", 1)[1]
        return value
    return None


def load_mapping(path: Path | None, entity_type: str) -> dict[str, str]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"mapping file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"mapping file has no header: {path}")
        # ``id1`` is the local integer key in the original TCM-suite v4
        # herb, compound, target-protein, and disease CSV files.
        id_candidates = (f"{entity_type}_id", "entity_id", "id", "id1")
        id_column = next((name for name in id_candidates if name in reader.fieldnames), None)
        if id_column is None:
            raise ValueError(
                f"{path}: expected one of the ID columns {', '.join(id_candidates)}"
            )
        mapping: dict[str, str] = {}
        for row in reader:
            entity_id = (row.get(id_column) or "").strip()
            label = _mapping_label(entity_type, row)
            if entity_id and label:
                mapping[entity_id] = label
        return mapping


def discover_mappings(dataset_dir: Path, explicit: dict[str, Path | None]) -> dict[str, dict[str, str]]:
    mappings: dict[str, dict[str, str]] = {}
    automatic_names = {
        "herb": ("herb_id_mapping.csv", "herb.csv"),
        "compound": ("compound_id_mapping.csv", "compound.csv"),
        "protein": ("protein_id_mapping.csv", "target_protein.csv"),
        "disease": ("disease_id_mapping.csv", "disease.csv"),
    }
    for entity_type in ENTITY_COLORS:
        candidate = explicit.get(entity_type)
        if candidate is None:
            candidate = next(
                (
                    dataset_dir / name
                    for name in automatic_names[entity_type]
                    if (dataset_dir / name).is_file()
                ),
                None,
            )
        mappings[entity_type] = load_mapping(candidate, entity_type)
    return mappings


def entity_label(
    entity_type: str,
    entity_id: str,
    mappings: dict[str, dict[str, str]],
    override: str | None = None,
) -> str:
    if override:
        return override
    mapped = mappings.get(entity_type, {}).get(entity_id)
    if mapped:
        return mapped
    return f"{entity_type.title()} {entity_id}"


def _retain_herb_coverage(
    ranked: Sequence[str],
    coverage: dict[str, set[str]],
    herb_ids: Sequence[str],
    limit: int,
) -> list[str]:
    """Retain high-ranked items while giving every reachable herb a path."""

    required: list[str] = []
    covered: set[str] = set()
    target = set(herb_ids)
    for value in ranked:
        newly_covered = coverage.get(value, set()) & target - covered
        if newly_covered:
            required.append(value)
            covered.update(coverage.get(value, set()))
        if covered >= target:
            break
    selected = required[:limit]
    for value in ranked:
        if len(selected) >= limit:
            break
        if value not in selected:
            selected.append(value)
    selected_set = set(selected)
    return [value for value in ranked if value in selected_set]


def build_case_graph(
    paths: dict[str, Path],
    mappings: dict[str, dict[str, str]],
    herb_ids: Sequence[str],
    disease_id: str | None,
    herb_labels: dict[str, str],
    disease_label: str | None,
    max_compounds: int,
    max_proteins: int,
    max_diseases: int,
) -> CaseGraph:
    herb_ids = list(dict.fromkeys(str(value) for value in herb_ids))
    if not herb_ids:
        raise ValueError("at least one herb ID is required for a case graph")
    herb_set = set(herb_ids)
    compounds_by_herb = {herb: set() for herb in herb_ids}
    herbs_by_compound: dict[str, set[str]] = {}
    for herb, compound in positive_rows(paths["H-C"]):
        if herb in herb_set:
            compounds_by_herb[herb].add(compound)
            herbs_by_compound.setdefault(compound, set()).add(herb)
    missing_herbs = [herb for herb in herb_ids if not compounds_by_herb[herb]]
    if missing_herbs:
        raise ValueError(
            "herb(s) with no positive H-C relation: " + ", ".join(missing_herbs)
        )

    compounds = set(herbs_by_compound)
    cp_edges = [
        (compound, protein)
        for compound, protein in positive_rows(paths["C-P"])
        if compound in compounds
    ]
    proteins = {protein for _, protein in cp_edges}
    if not proteins:
        raise ValueError(
            "selected herb(s) have no H-C-P path: " + ", ".join(herb_ids)
        )

    pd_edges = [(protein, disease) for protein, disease in positive_rows(paths["P-D"]) if protein in proteins]
    direct_diseases_by_herb = {herb: set() for herb in herb_ids}
    for herb, disease in positive_rows(paths["H-D"]):
        if herb in herb_set:
            direct_diseases_by_herb[herb].add(disease)
    direct_diseases = set().union(*direct_diseases_by_herb.values())

    compounds_by_protein: dict[str, set[str]] = {}
    for compound, protein in cp_edges:
        compounds_by_protein.setdefault(protein, set()).add(compound)
    herbs_by_protein = {
        protein: set().union(
            *(herbs_by_compound[compound] for compound in protein_compounds)
        )
        for protein, protein_compounds in compounds_by_protein.items()
    }
    proteins_by_disease: dict[str, set[str]] = {}
    for protein, disease in pd_edges:
        proteins_by_disease.setdefault(disease, set()).add(protein)

    support: dict[str, int] = {}
    path_herb_coverage: dict[str, set[str]] = {}
    direct_herb_coverage: dict[str, set[str]] = {}
    for candidate, candidate_proteins in proteins_by_disease.items():
        support[candidate] = sum(len(compounds_by_protein[p]) for p in candidate_proteins)
        path_herb_coverage[candidate] = set().union(
            *(herbs_by_protein[protein] for protein in candidate_proteins)
        )
        direct_herb_coverage[candidate] = {
            herb
            for herb in herb_ids
            if candidate in direct_diseases_by_herb[herb]
        }

    if disease_id is not None:
        if disease_id not in proteins_by_disease:
            raise ValueError(
                f"disease {disease_id} has no H-C-P-D path from the selected herbs"
            )
        selected_diseases = [disease_id]
        selection_note = "User-selected disease; intermediate nodes ranked by retained path support."
    else:
        candidates = [candidate for candidate in support if candidate in direct_diseases]
        if not candidates:
            candidates = list(support)
        selected_diseases = sorted(
            candidates,
            key=lambda value: (
                -len(direct_herb_coverage[value]),
                -len(path_herb_coverage[value]),
                -support[value],
                value,
            ),
        )[:max_diseases]
        selection_note = (
            "Diseases ranked by target-herb coverage and recorded H-C-P-D path support; "
            "candidates are restricted to recorded H-D associations when available."
        )

    selected_disease_set = set(selected_diseases)
    protein_disease_counts: dict[str, int] = {}
    for protein, disease in pd_edges:
        if disease in selected_disease_set:
            protein_disease_counts[protein] = protein_disease_counts.get(protein, 0) + 1
    ranked_proteins = sorted(
        protein_disease_counts,
        key=lambda value: (
            -len(herbs_by_protein.get(value, set())),
            -protein_disease_counts[value],
            -len(compounds_by_protein.get(value, set())),
            value,
        ),
    )
    selected_proteins = _retain_herb_coverage(
        ranked_proteins, herbs_by_protein, herb_ids, max_proteins
    )
    selected_protein_set = set(selected_proteins)

    compound_scores: dict[str, int] = {}
    for compound, protein in cp_edges:
        if protein in selected_protein_set:
            compound_scores[compound] = compound_scores.get(compound, 0) + 1
    ranked_compounds = sorted(
        compound_scores,
        key=lambda value: (
            -len(herbs_by_compound.get(value, set())),
            -compound_scores[value],
            value,
        ),
    )
    selected_compounds = _retain_herb_coverage(
        ranked_compounds, herbs_by_compound, herb_ids, max_compounds
    )
    selected_compound_set = set(selected_compounds)

    nodes = [
        Node(
            key=f"h:{herb_id}",
            entity_type="herb",
            entity_id=herb_id,
            label=entity_label("herb", herb_id, mappings, herb_labels.get(herb_id)),
        )
        for herb_id in herb_ids
    ]
    nodes.extend(
        Node(f"c:{value}", "compound", value, entity_label("compound", value, mappings))
        for value in selected_compounds
    )
    nodes.extend(
        Node(f"p:{value}", "protein", value, entity_label("protein", value, mappings))
        for value in selected_proteins
    )
    nodes.extend(
        Node(
            f"d:{value}",
            "disease",
            value,
            entity_label(
                "disease",
                value,
                mappings,
                disease_label if disease_id == value else None,
            ),
        )
        for value in selected_diseases
    )

    edges = [
        Edge(f"h:{herb}", f"c:{compound}", "H-C")
        for herb in herb_ids
        for compound in selected_compounds
        if compound in compounds_by_herb[herb]
    ]
    edges.extend(
        Edge(f"c:{compound}", f"p:{protein}", "C-P")
        for compound, protein in cp_edges
        if compound in selected_compound_set and protein in selected_protein_set
    )
    edges.extend(
        Edge(f"p:{protein}", f"d:{disease}", "P-D")
        for protein, disease in pd_edges
        if protein in selected_protein_set and disease in selected_disease_set
    )
    edges.extend(
        Edge(f"h:{herb}", f"d:{disease}", "H-D")
        for herb in herb_ids
        for disease in selected_diseases
        if disease in direct_diseases_by_herb[herb]
    )

    return CaseGraph(
        herb_ids=herb_ids,
        disease_ids=selected_diseases,
        nodes=nodes,
        edges=edges,
        available_counts={
            "herb_compounds": len(compounds),
            "herb_compounds_by_herb": {
                herb: len(compounds_by_herb[herb]) for herb in herb_ids
            },
            "compound_protein_edges": len(cp_edges),
            "reachable_proteins": len(proteins),
            "protein_disease_edges": len(pd_edges),
            "reachable_diseases": len(proteins_by_disease),
            "recorded_herb_diseases": len(direct_diseases),
            "recorded_herb_diseases_by_herb": {
                herb: len(direct_diseases_by_herb[herb]) for herb in herb_ids
            },
        },
        selection_note=selection_note,
    )


class Scene:
    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.items: list[dict[str, object]] = []

    def add(self, kind: str, **values: object) -> None:
        self.items.append({"kind": kind, **values})


def _mix(color: str, other: str = "#FFFFFF", amount: float = 0.5) -> str:
    left = tuple(int(color[index:index + 2], 16) for index in (1, 3, 5))
    right = tuple(int(other[index:index + 2], 16) for index in (1, 3, 5))
    mixed = tuple(round(a * (1 - amount) + b * amount) for a, b in zip(left, right))
    return "#" + "".join(f"{value:02X}" for value in mixed)


def _positions(count: int, top: float, bottom: float) -> list[float]:
    if count <= 1:
        return [(top + bottom) / 2]
    return [top + index * (bottom - top) / (count - 1) for index in range(count)]


def _shorten(value: str, length: int = 28) -> str:
    value = " ".join(value.split())
    return value if len(value) <= length else value[: length - 1].rstrip() + "…"


def _display_label(node: Node) -> str:
    """Return a concise visible label while preserving the full mapped label."""

    value = node.label
    if node.entity_type == "disease" and "|" in value:
        value = value.split("|", 1)[0].strip()
    return _shorten(value)


def build_case_scene(graph: CaseGraph, title: str) -> tuple[Scene, dict[str, tuple[float, float]]]:
    scene = Scene(1600, 920)
    scene.add("rect", x=0, y=0, width=1600, height=920, fill="#FFFFFF", stroke="#FFFFFF", radius=0)
    scene.add("text", x=70, y=70, text=title, size=32, color="#20313B", weight="bold", anchor="start")
    scene.add(
        "text", x=70, y=106,
        text="Recorded database relations; deterministic path-support filtering for readability",
        size=16, color="#667780", weight="normal", anchor="start",
    )

    columns = {"herb": 180, "compound": 555, "protein": 965, "disease": 1380}
    headings = {
        "herb": "Herbs" if len(graph.herb_ids) > 1 else "Herb",
        "compound": "Compounds",
        "protein": "Proteins",
        "disease": "Diseases",
    }
    for entity_type, x in columns.items():
        scene.add("text", x=x, y=162, text=headings[entity_type], size=21, color="#263942", weight="bold", anchor="middle")
        scene.add("line", x1=x - 112, y1=179, x2=x + 112, y2=179, stroke=ENTITY_COLORS[entity_type][0], width=3)

    by_type: dict[str, list[Node]] = {name: [] for name in columns}
    for node in graph.nodes:
        by_type[node.entity_type].append(node)
    locations: dict[str, tuple[float, float]] = {}
    for entity_type, nodes in by_type.items():
        ys = _positions(len(nodes), 260, 720)
        for node, y in zip(nodes, ys):
            locations[node.key] = (columns[entity_type], y)

    for edge in graph.edges:
        x1, y1 = locations[edge.source]
        x2, y2 = locations[edge.target]
        relation_color = _mix(RELATION_COLORS[edge.relation], amount=0.44)
        if edge.relation == "H-D":
            scene.add(
                "path",
                points=(x1 + 112, y1, 590, 190, 1040, 190, x2 - 112, y2),
                stroke=RELATION_COLORS[edge.relation], width=2.4, dash="10 8",
                source=edge.source, target=edge.target, relation=edge.relation,
            )
        else:
            bend = max(70, (x2 - x1) * 0.38)
            scene.add(
                "path",
                points=(x1 + 112, y1, x1 + bend, y1, x2 - bend, y2, x2 - 112, y2),
                stroke=relation_color, width=1.7, dash="",
                source=edge.source, target=edge.target, relation=edge.relation,
            )

    for node in graph.nodes:
        x, y = locations[node.key]
        border, fill = ENTITY_COLORS[node.entity_type]
        scene.add(
            "rect", x=x - 112, y=y - 31, width=224, height=62,
            fill=fill, stroke=border, stroke_width=2.2, radius=20, node=node.key,
            label=node.label, entity_type=node.entity_type, entity_id=node.entity_id,
        )
        scene.add(
            "circle", cx=x - 86, cy=y, radius=8, fill=border, stroke=border,
            stroke_width=1, node=node.key, label=node.label,
            entity_type=node.entity_type, entity_id=node.entity_id,
        )
        scene.add(
            "text", x=x + 8, y=y + 6, text=_display_label(node), size=15,
            color="#25343B", weight="bold", anchor="middle", node=node.key,
            label=node.label, entity_type=node.entity_type, entity_id=node.entity_id,
        )

    legend_y = 826
    scene.add("rect", x=56, y=785, width=1488, height=92, fill="#F7F9FA", stroke="#DDE5E8", stroke_width=1.5, radius=18)
    scene.add("text", x=82, y=821, text="Relations", size=16, color="#344850", weight="bold", anchor="start")
    legend_x = 205
    for relation in ("H-C", "C-P", "P-D", "H-D"):
        scene.add("line", x1=legend_x, y1=legend_y, x2=legend_x + 58, y2=legend_y, stroke=RELATION_COLORS[relation], width=3, dash="8 6" if relation == "H-D" else "")
        scene.add("text", x=legend_x + 68, y=legend_y + 6, text=relation, size=15, color="#42545C", weight="normal", anchor="start")
        legend_x += 145
    scene.add(
        "text", x=790, y=legend_y + 6,
        text="Dashed H-D edges are direct records; solid edges show H-C-P-D context.",
        size=14, color="#667780", weight="normal", anchor="start",
    )
    return scene, locations


def build_overview_scene(inventory: dict[str, object]) -> Scene:
    scene = Scene(1400, 650)
    scene.add("rect", x=0, y=0, width=1400, height=650, fill="#FFFFFF", stroke="#FFFFFF", radius=0)
    scene.add("text", x=70, y=66, text="ASTRA database relationship inventory", size=31, color="#20313B", weight="bold", anchor="start")
    scene.add("text", x=70, y=103, text="All positive relation rows are counted; individual edges are intentionally not overplotted.", size=16, color="#667780", weight="normal", anchor="start")
    xs = {"herb": 180, "compound": 520, "protein": 880, "disease": 1220}
    labels = {"herb": "Herbs", "compound": "Compounds", "protein": "Proteins", "disease": "Diseases"}
    entity_counts = inventory["entities"]
    relation_counts = inventory["relations"]
    for entity_type, x in xs.items():
        border, fill = ENTITY_COLORS[entity_type]
        scene.add("rect", x=x - 125, y=225, width=250, height=190, fill=fill, stroke=border, stroke_width=2.5, radius=28)
        scene.add("circle", cx=x, cy=275, radius=19, fill=border, stroke=border, stroke_width=1)
        scene.add("text", x=x, y=337, text=labels[entity_type], size=23, color="#263942", weight="bold", anchor="middle")
        scene.add("text", x=x, y=378, text=f"{entity_counts[entity_type]:,} entities", size=18, color="#52656D", weight="normal", anchor="middle")
    for relation, left, right, y in (
        ("H-C", "herb", "compound", 310),
        ("C-P", "compound", "protein", 310),
        ("P-D", "protein", "disease", 310),
    ):
        x1, x2 = xs[left] + 128, xs[right] - 128
        scene.add("line", x1=x1, y1=y, x2=x2, y2=y, stroke=RELATION_COLORS[relation], width=4)
        scene.add("text", x=(x1 + x2) / 2, y=y - 18, text=f"{relation}: {relation_counts[relation]:,}", size=16, color="#40535B", weight="bold", anchor="middle")
    scene.add(
        "path", points=(xs["herb"], 222, 470, 120, 930, 120, xs["disease"], 222),
        stroke=RELATION_COLORS["H-D"], width=3.2, dash="10 8",
    )
    scene.add("text", x=700, y=142, text=f"H-D: {relation_counts['H-D']:,}", size=17, color=RELATION_COLORS["H-D"], weight="bold", anchor="middle")
    scene.add("rect", x=70, y=500, width=1260, height=90, fill="#F7F9FA", stroke="#DDE5E8", stroke_width=1.4, radius=18)
    scene.add("text", x=700, y=540, text="This overview covers all four relation types used by ASTRA.", size=18, color="#344850", weight="bold", anchor="middle")
    scene.add("text", x=700, y=570, text="Use a herb-centred case view to inspect individual recorded paths without producing an unreadable graph.", size=15, color="#667780", weight="normal", anchor="middle")
    return scene


def _svg_attributes(item: dict[str, object], keys: Iterable[str]) -> str:
    values = []
    for key in keys:
        if key in item and item[key] not in (None, ""):
            attribute = key.replace("_", "-")
            values.append(f' data-{attribute}="{escape(str(item[key]), quote=True)}"')
    return "".join(values)


def scene_to_svg(scene: Scene) -> str:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {scene.width:g} {scene.height:g}" role="img">',
        '<style>text{font-family:Arial,Helvetica,sans-serif}.node-shape,.node-text{cursor:pointer;transition:opacity .15s}.edge{transition:opacity .15s}</style>',
    ]
    for item in scene.items:
        kind = item["kind"]
        data = _svg_attributes(
            item,
            ("node", "source", "target", "relation", "label", "entity_type", "entity_id"),
        )
        tooltip = ""
        if "node" in item:
            label = escape(str(item.get("label", item["node"])))
            entity_type = escape(str(item.get("entity_type", "entity")).title())
            entity_id = escape(str(item.get("entity_id", "")))
            tooltip = f"<title>{label} ({entity_type} ID {entity_id})</title>"
        if kind == "rect":
            classes = "node-shape" if "node" in item else ""
            parts.append(
                f'<rect class="{classes}" x="{item["x"]}" y="{item["y"]}" width="{item["width"]}" height="{item["height"]}" '
                f'rx="{item.get("radius", 0)}" fill="{item["fill"]}" stroke="{item["stroke"]}" stroke-width="{item.get("stroke_width", 1)}"{data}>{tooltip}</rect>'
            )
        elif kind == "circle":
            classes = "node-shape" if "node" in item else ""
            parts.append(
                f'<circle class="{classes}" cx="{item["cx"]}" cy="{item["cy"]}" r="{item["radius"]}" fill="{item["fill"]}" stroke="{item["stroke"]}" stroke-width="{item.get("stroke_width", 1)}"{data}>{tooltip}</circle>'
            )
        elif kind == "line":
            dash = f' stroke-dasharray="{item["dash"]}"' if item.get("dash") else ""
            parts.append(
                f'<line x1="{item["x1"]}" y1="{item["y1"]}" x2="{item["x2"]}" y2="{item["y2"]}" stroke="{item["stroke"]}" stroke-width="{item.get("width", 1)}" stroke-linecap="round"{dash}{data}/>'
            )
        elif kind == "path":
            x1, y1, c1x, c1y, c2x, c2y, x2, y2 = item["points"]
            dash = f' stroke-dasharray="{item["dash"]}"' if item.get("dash") else ""
            parts.append(
                f'<path class="edge" d="M{x1},{y1} C{c1x},{c1y} {c2x},{c2y} {x2},{y2}" fill="none" stroke="{item["stroke"]}" stroke-width="{item.get("width", 1)}" stroke-linecap="round"{dash}{data}/>'
            )
        elif kind == "text":
            classes = "node-text" if "node" in item else ""
            parts.append(
                f'<text class="{classes}" x="{item["x"]}" y="{item["y"]}" font-size="{item["size"]}" fill="{item["color"]}" font-weight="{item["weight"]}" text-anchor="{item["anchor"]}"{data}>{escape(str(item["text"]))}{tooltip}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _rgb(color: str) -> tuple[float, float, float]:
    return tuple(int(color[index:index + 2], 16) / 255 for index in (1, 3, 5))  # type: ignore[return-value]


def _pdf_text(value: str) -> bytes:
    normalized = value.replace("–", "-").replace("—", "-").replace("…", "...")
    encoded = normalized.encode("cp1252", errors="replace")
    return encoded.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")


def _rounded_rect_commands(x: float, y: float, width: float, height: float, radius: float, page_height: float) -> str:
    y_pdf = page_height - y - height
    r = min(radius, width / 2, height / 2)
    k = 0.5522847498
    return (
        f"{x+r:.2f} {y_pdf:.2f} m {x+width-r:.2f} {y_pdf:.2f} l "
        f"{x+width-r+k*r:.2f} {y_pdf:.2f} {x+width:.2f} {y_pdf+r-k*r:.2f} {x+width:.2f} {y_pdf+r:.2f} c "
        f"{x+width:.2f} {y_pdf+height-r:.2f} l "
        f"{x+width:.2f} {y_pdf+height-r+k*r:.2f} {x+width-r+k*r:.2f} {y_pdf+height:.2f} {x+width-r:.2f} {y_pdf+height:.2f} c "
        f"{x+r:.2f} {y_pdf+height:.2f} l "
        f"{x+r-k*r:.2f} {y_pdf+height:.2f} {x:.2f} {y_pdf+height-r+k*r:.2f} {x:.2f} {y_pdf+height-r:.2f} c "
        f"{x:.2f} {y_pdf+r:.2f} l "
        f"{x:.2f} {y_pdf+r-k*r:.2f} {x+r-k*r:.2f} {y_pdf:.2f} {x+r:.2f} {y_pdf:.2f} c h"
    )


def scene_to_pdf(scene: Scene) -> bytes:
    commands: list[bytes] = []
    height = scene.height
    for item in scene.items:
        kind = item["kind"]
        if kind in {"rect", "circle"}:
            fill = _rgb(str(item["fill"]))
            stroke = _rgb(str(item["stroke"]))
            stroke_width = float(item.get("stroke_width", 1))
            commands.append(f"{fill[0]:.4f} {fill[1]:.4f} {fill[2]:.4f} rg {stroke[0]:.4f} {stroke[1]:.4f} {stroke[2]:.4f} RG {stroke_width:.2f} w [] 0 d\n".encode())
            if kind == "rect":
                path = _rounded_rect_commands(float(item["x"]), float(item["y"]), float(item["width"]), float(item["height"]), float(item.get("radius", 0)), height)
            else:
                cx, cy, radius = float(item["cx"]), height - float(item["cy"]), float(item["radius"])
                k = radius * 0.5522847498
                path = (
                    f"{cx+radius:.2f} {cy:.2f} m {cx+radius:.2f} {cy+k:.2f} {cx+k:.2f} {cy+radius:.2f} {cx:.2f} {cy+radius:.2f} c "
                    f"{cx-k:.2f} {cy+radius:.2f} {cx-radius:.2f} {cy+k:.2f} {cx-radius:.2f} {cy:.2f} c "
                    f"{cx-radius:.2f} {cy-k:.2f} {cx-k:.2f} {cy-radius:.2f} {cx:.2f} {cy-radius:.2f} c "
                    f"{cx+k:.2f} {cy-radius:.2f} {cx+radius:.2f} {cy-k:.2f} {cx+radius:.2f} {cy:.2f} c h"
                )
            commands.append((path + " B\n").encode())
        elif kind in {"line", "path"}:
            stroke = _rgb(str(item["stroke"]))
            width = float(item.get("width", 1))
            dash = str(item.get("dash", "")).split()
            dash_command = "[" + " ".join(dash) + "] 0 d" if dash else "[] 0 d"
            commands.append(f"{stroke[0]:.4f} {stroke[1]:.4f} {stroke[2]:.4f} RG {width:.2f} w {dash_command}\n".encode())
            if kind == "line":
                path = f"{float(item['x1']):.2f} {height-float(item['y1']):.2f} m {float(item['x2']):.2f} {height-float(item['y2']):.2f} l S\n"
            else:
                x1, y1, c1x, c1y, c2x, c2y, x2, y2 = (float(value) for value in item["points"])
                path = f"{x1:.2f} {height-y1:.2f} m {c1x:.2f} {height-c1y:.2f} {c2x:.2f} {height-c2y:.2f} {x2:.2f} {height-y2:.2f} c S\n"
            commands.append(path.encode())
        elif kind == "text":
            color = _rgb(str(item["color"]))
            size = float(item["size"])
            value = str(item["text"])
            estimated_width = len(value.replace("…", "...")) * size * 0.52
            x = float(item["x"])
            if item["anchor"] == "middle":
                x -= estimated_width / 2
            elif item["anchor"] == "end":
                x -= estimated_width
            y = height - float(item["y"])
            font = "F2" if item["weight"] == "bold" else "F1"
            commands.append(
                f"{color[0]:.4f} {color[1]:.4f} {color[2]:.4f} rg BT /{font} {size:.2f} Tf {x:.2f} {y:.2f} Td (".encode()
                + _pdf_text(value)
                + b") Tj ET\n"
            )

    content = b"".join(commands)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {scene.width:g} {scene.height:g}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>".encode(),
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode())
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref = len(output)
    output.extend(f"xref\n0 {len(objects)+1}\n".encode())
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode())
    output.extend(f"trailer\n<< /Size {len(objects)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(output)


def interactive_html(svg: str, title: str, nodes: Sequence[Node] = ()) -> str:
    options = "".join(
        f'<option value="{escape(node.key, quote=True)}">{escape(node.label)} ({escape(node.entity_type)})</option>'
        for node in nodes
    )
    controls = (
        '<label for="node-search">Highlight node</label>'
        f'<select id="node-search"><option value="">Show all</option>{options}</select>'
        if nodes else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)}</title>
<style>body{{margin:0;background:#edf2f4;color:#20313b;font-family:Arial,sans-serif}}main{{max-width:1500px;margin:24px auto;background:white;border-radius:18px;box-shadow:0 12px 40px #20313b22;overflow:hidden}}.controls{{padding:16px 24px;border-bottom:1px solid #dce5e8;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}select{{padding:8px 12px;border:1px solid #b8c7cd;border-radius:8px;background:white}}#node-detail{{margin-left:auto;color:#52656d}}svg{{display:block;width:100%;height:auto}}</style></head>
<body><main><div class="controls">{controls}<span>Click a node to isolate its recorded neighbors.</span><span id="node-detail">Hover over a node for its full name and ID.</span></div>{svg}</main>
<script>
const svg=document.querySelector('svg'), select=document.querySelector('#node-search'), detail=document.querySelector('#node-detail');
function highlight(key){{
  const edges=[...svg.querySelectorAll('.edge')], connected=new Set([key]);
  edges.forEach(e=>{{const active=!key||e.dataset.source===key||e.dataset.target===key;e.style.opacity=active?'1':'0.07';if(active){{connected.add(e.dataset.source);connected.add(e.dataset.target)}}}});
  svg.querySelectorAll('[data-node]').forEach(n=>n.style.opacity=!key||connected.has(n.dataset.node)?'1':'0.22');
  const node=key?svg.querySelector(`[data-node="${{CSS.escape(key)}}"]`):null;
  detail.textContent=node?`${{node.dataset.label}} (${{node.dataset.entityType}} ID ${{node.dataset.entityId}})`:'Hover over a node for its full name and ID.';
}}
svg.addEventListener('click',event=>{{const target=event.target.closest('[data-node]');highlight(target?target.dataset.node:'');if(select)select.value=target?target.dataset.node:'';}});
svg.addEventListener('mouseover',event=>{{const target=event.target.closest('[data-node]');if(target)detail.textContent=`${{target.dataset.label}} (${{target.dataset.entityType}} ID ${{target.dataset.entityId}})`;}});
if(select)select.addEventListener('change',()=>highlight(select.value));
</script></body></html>"""


def write_scene(scene: Scene, stem: Path, title: str, nodes: Sequence[Node] = ()) -> dict[str, str]:
    svg = scene_to_svg(scene)
    svg_path = stem.with_suffix(".svg")
    pdf_path = stem.with_suffix(".pdf")
    html_path = stem.with_suffix(".html")
    svg_path.write_text(svg, encoding="utf-8")
    pdf_path.write_bytes(scene_to_pdf(scene))
    html_path.write_text(interactive_html(svg, title, nodes), encoding="utf-8")
    return {"svg": str(svg_path), "pdf": str(pdf_path), "html": str(html_path)}


def _split_repeated_values(values: Sequence[str] | None) -> list[str]:
    result: list[str] = []
    for value in values or ():
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return list(dict.fromkeys(result))


def _parse_herb_labels(
    herb_ids: Sequence[str], values: Sequence[str] | None
) -> dict[str, str]:
    labels: dict[str, str] = {}
    positional: list[str] = []
    for value in values or ():
        if "=" in value:
            herb_id, label = value.split("=", 1)
            herb_id, label = herb_id.strip(), label.strip()
            if herb_id not in herb_ids:
                raise ValueError(
                    f"herb label refers to unselected herb ID {herb_id!r}"
                )
            if not label:
                raise ValueError(f"empty label for herb ID {herb_id!r}")
            labels[herb_id] = label
        else:
            positional.append(value.strip())
    unlabeled = [herb_id for herb_id in herb_ids if herb_id not in labels]
    if len(positional) > len(unlabeled):
        raise ValueError("more positional herb labels than selected herb IDs")
    labels.update(zip(unlabeled, positional))
    return labels


def _safe_filename_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
    return token or "entity"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize all ASTRA relation types and an auditable herb-centred subgraph."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--herb-id",
        action="append",
        help="Herb ID; repeat the option or use comma-separated IDs for comparison.",
    )
    parser.add_argument("--disease-id")
    parser.add_argument(
        "--herb-label",
        action="append",
        help="Optional label in herb order or as ID=Label; may be repeated.",
    )
    parser.add_argument("--disease-label")
    parser.add_argument("--max-compounds", type=int, default=8)
    parser.add_argument("--max-proteins", type=int, default=8)
    parser.add_argument("--max-diseases", type=int, default=6)
    parser.add_argument("--herb-mapping", type=Path)
    parser.add_argument("--compound-mapping", type=Path)
    parser.add_argument("--protein-mapping", type=Path)
    parser.add_argument("--disease-mapping", type=Path)
    parser.add_argument("--title", default="Recorded multirelational context in ASTRA")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if min(args.max_compounds, args.max_proteins, args.max_diseases) < 1:
        print("max node counts must be positive", file=sys.stderr)
        return 2
    try:
        dataset_dir = args.dataset_dir.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = require_relation_files(dataset_dir)
        inventory = database_inventory(paths)
        mappings = discover_mappings(
            dataset_dir,
            {
                "herb": args.herb_mapping,
                "compound": args.compound_mapping,
                "protein": args.protein_mapping,
                "disease": args.disease_mapping,
            },
        )

        outputs: dict[str, object] = {
            "overview": write_scene(
                build_overview_scene(inventory),
                output_dir / "astra_relationship_inventory",
                "ASTRA database relationship inventory",
            )
        }
        manifest: dict[str, object] = {
            "dataset_dir": str(dataset_dir),
            "inventory": inventory,
            "outputs": outputs,
        }
        manifest_name = "visualization_manifest.json"
        herb_ids = _split_repeated_values(args.herb_id)
        herb_labels = _parse_herb_labels(herb_ids, args.herb_label)
        if herb_ids:
            graph = build_case_graph(
                paths=paths,
                mappings=mappings,
                herb_ids=herb_ids,
                disease_id=str(args.disease_id) if args.disease_id else None,
                herb_labels=herb_labels,
                disease_label=args.disease_label,
                max_compounds=args.max_compounds,
                max_proteins=args.max_proteins,
                max_diseases=args.max_diseases,
            )
            scene, _ = build_case_scene(graph, args.title)
            identifier = "_".join(_safe_filename_token(value) for value in herb_ids)
            case_name = "astra_case_herb" if len(herb_ids) == 1 else "astra_case_herbs"
            stem = output_dir / f"{case_name}_{identifier}"
            manifest_name = f"visualization_manifest_{case_name.removeprefix('astra_case_')}_{identifier}.json"
            outputs["case"] = write_scene(scene, stem, args.title, graph.nodes)
            manifest["case"] = {
                "herb_id": graph.herb_ids[0] if len(graph.herb_ids) == 1 else None,
                "herb_ids": graph.herb_ids,
                "disease_ids": graph.disease_ids,
                "available_counts": graph.available_counts,
                "selection_note": graph.selection_note,
                "displayed_nodes": [node.__dict__ for node in graph.nodes],
                "displayed_edges": [edge.__dict__ for edge in graph.edges],
                "limits": {
                    "max_compounds": args.max_compounds,
                    "max_proteins": args.max_proteins,
                    "max_diseases": args.max_diseases,
                },
            }
        serialized_manifest = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
        manifest_path = output_dir / manifest_name
        manifest_path.write_text(serialized_manifest, encoding="utf-8")
        latest_manifest = output_dir / "visualization_manifest.json"
        if latest_manifest != manifest_path:
            latest_manifest.write_text(serialized_manifest, encoding="utf-8")
        print(json.dumps({"manifest": str(manifest_path), "outputs": outputs}, indent=2))
        return 0
    except (FileNotFoundError, ValueError) as error:
        print(f"astra-visualize: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
