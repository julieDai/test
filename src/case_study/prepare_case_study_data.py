#!/usr/bin/env python3
# coding: utf-8
"""Prepare target-herb withheld-association data for the HerbHG case study."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


CASE_STUDY_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CASE_STUDY_DIR.parent
PROJECT_DIR = SOURCE_DIR.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from herbhg.experiments.runtime import activate_tcm_backend

activate_tcm_backend(SOURCE_DIR)

from util.paths import first_environment_value, resolve_path


HERB_IDS = {'ChuanXiong': 891, 'HuangQi': 2565}


def _command_path(value: str | os.PathLike[str]) -> Path:
    return resolve_path(value, Path.cwd())


def _environment_path(names, default: Path) -> Path:
    value = first_environment_value(names)
    return resolve_path(value, PROJECT_DIR) if value else default.resolve(strict=False)


def prepare_case_study_data(
    target_herb_id: int,
    herb_name: str,
    *,
    source_data_dir: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    seed: int = 42,
):
    """Remove one herb's H--D records and build a balanced training file."""

    print(f"\n{'=' * 80}")
    print(f"Preparing HerbHG case-study data for {herb_name} (ID={target_herb_id})")
    print(f"{'=' * 80}\n")

    if source_data_dir is None:
        source_dir = _environment_path(
            ('HERBHG_DATA_DIR', 'ASTRA_DATA_DIR', 'CDRUG_DATA_DIR'),
            PROJECT_DIR / 'data' / 'tcm_suite',
        )
    else:
        source_dir = _command_path(source_data_dir)
    if output_dir is None:
        destination = _environment_path(
            ('HERBHG_CASE_DATA_DIR', 'ASTRA_CASE_DATA_DIR', 'CDRUG_CASE_DATA_DIR'),
            PROJECT_DIR / 'data' / 'processed' / 'case_study',
        )
    else:
        destination = _command_path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    h_d_path = source_dir / 'H_D.txt'
    negative_path = source_dir / 'negativehdlist.txt'
    if not h_d_path.is_file():
        raise FileNotFoundError(f'H--D 数据不存在: {h_d_path}')
    if not negative_path.is_file():
        raise FileNotFoundError(f'负样本数据不存在: {negative_path}')

    output_h_d = destination / f'H_D_{herb_name}.txt'
    output_info = destination / f'{herb_name}_removed_pairs.txt'

    print(f"Step 1: 读取 {h_d_path} 并移除 {herb_name} 的配对...")
    positive_pairs = []
    removed_pairs = []
    removed_diseases = set()
    with h_d_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            herb_id, disease_id, label = map(int, parts[:3])
            if herb_id == target_herb_id:
                removed_pairs.append((herb_id, disease_id, label))
                removed_diseases.add(disease_id)
            else:
                positive_pairs.append((herb_id, disease_id, label))

    print(f"  - 原始正样本数量: {len(positive_pairs) + len(removed_pairs)}")
    print(f"  - 移除的 {herb_name} 配对数量: {len(removed_pairs)}")
    print(f"  - 涉及的疾病数量: {len(removed_diseases)}")
    print(f"  - 剩余正样本数量: {len(positive_pairs)}")

    with output_info.open('w', encoding='utf-8') as handle:
        handle.write(f"# {herb_name} (ID={target_herb_id}) 移除的草药-疾病配对\n")
        handle.write("# 格式: 草药ID 疾病ID 标签\n")
        handle.write(f"# 总计: {len(removed_pairs)} 个配对\n\n")
        for herb_id, disease_id, label in removed_pairs:
            handle.write(f"{herb_id}\t{disease_id}\t{label}\n")
    print(f"  - 已保存移除的配对信息到: {output_info}")

    print(f"\nStep 2: 读取负样本并移除包含 {herb_name} 的样本...")
    negative_candidates = []
    with negative_path.open('r', encoding='utf-8') as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            herb_id, disease_id, label = map(int, parts[:3])
            if herb_id != target_herb_id:
                negative_candidates.append((herb_id, disease_id, label))
    print(f"  - 可用负样本候选数量: {len(negative_candidates)}")

    num_positive = len(positive_pairs)
    print(f"\nStep 3: 采样 {num_positive} 个负样本...")
    random_generator = random.Random(seed)
    if len(negative_candidates) < num_positive:
        print(
            f"  警告: 负样本候选数量 ({len(negative_candidates)}) "
            f"少于正样本数量 ({num_positive})，将使用全部候选"
        )
        sampled_negatives = negative_candidates
    else:
        sampled_negatives = random_generator.sample(negative_candidates, num_positive)
    print(f"  - 实际采样负样本数量: {len(sampled_negatives)}")

    print("\nStep 4: 合并正负样本并保存...")
    all_samples = positive_pairs + sampled_negatives
    random_generator.shuffle(all_samples)
    with output_h_d.open('w', encoding='utf-8') as handle:
        for herb_id, disease_id, label in all_samples:
            handle.write(f"{herb_id}\t{disease_id}\t{label}\n")

    print(
        f"  - 总样本数: {len(all_samples)} "
        f"(正样本: {len(positive_pairs)}, 负样本: {len(sampled_negatives)})"
    )
    print(f"  - 训练文件: {output_h_d}")
    return {
        'herb_id': target_herb_id,
        'herb_name': herb_name,
        'removed_pairs': len(removed_pairs),
        'removed_diseases': sorted(removed_diseases),
        'positive_samples': len(positive_pairs),
        'negative_samples': len(sampled_negatives),
        'total_samples': len(all_samples),
        'output_file': str(output_h_d),
        'info_file': str(output_info),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Prepare HerbHG withheld-association data')
    parser.add_argument(
        '--herb', default='both', choices=['ChuanXiong', 'HuangQi', 'both']
    )
    parser.add_argument(
        '--data-dir', help='包含 H_D.txt 和 negativehdlist.txt 的源数据目录。'
    )
    parser.add_argument(
        '--output-dir', help='Case-study data output directory.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=int(
            first_environment_value(('HERBHG_SEED', 'ASTRA_SEED', 'CDRUG_SEED'))
            or 42
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    herbs = list(HERB_IDS) if args.herb == 'both' else [args.herb]
    for herb_name in herbs:
        prepare_case_study_data(
            HERB_IDS[herb_name],
            herb_name,
            source_data_dir=args.data_dir,
            output_dir=args.output_dir,
            seed=args.seed,
        )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
