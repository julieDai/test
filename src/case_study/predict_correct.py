#!/usr/bin/env python3
# coding: utf-8
"""Rank diseases from embeddings produced by a HerbHG case-study run."""

from __future__ import annotations

import argparse
import json
import os
import pickle
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


def predict_final_correct(
    herb_name: str,
    target_herb_id: int,
    *,
    result_root: str | os.PathLike[str] | None = None,
    data_dir: str | os.PathLike[str] | None = None,
):
    """Score all represented diseases and evaluate recovery of withheld pairs."""

    import numpy as np
    import pandas as pd

    print(f"\n{'=' * 80}")
    print(f"HerbHG case-study prediction - {herb_name} (ID={target_herb_id})")
    print(f"{'=' * 80}\n")

    if result_root is None:
        resolved_result_root = _environment_path(
            ('HERBHG_CASE_RESULT_DIR', 'ASTRA_CASE_RESULT_DIR', 'CDRUG_CASE_RESULT_DIR'),
            PROJECT_DIR / 'outputs' / 'case_study',
        )
    else:
        resolved_result_root = _command_path(result_root)
    if data_dir is None:
        resolved_data_dir = _environment_path(
            ('HERBHG_CASE_DATA_DIR', 'ASTRA_CASE_DATA_DIR', 'CDRUG_CASE_DATA_DIR'),
            PROJECT_DIR / 'data' / 'processed' / 'case_study',
        )
    else:
        resolved_data_dir = _command_path(data_dir)
    result_dir = resolved_result_root / f'results_{herb_name}'

    print("Step 1: load HerbHG embeddings...")
    herb_emb_file = result_dir / 'herb_embeddings.npy'
    disease_emb_file = result_dir / 'disease_embeddings.npy'
    id_mapping_file = result_dir / 'id_mapping.pkl'
    missing = [
        path
        for path in (herb_emb_file, disease_emb_file, id_mapping_file)
        if not path.is_file()
    ]
    if missing:
        print("错误: 缺少训练输出:")
        for path in missing:
            print(f"  - {path}")
        print("请先运行 train_final.py")
        return None

    herb_embeddings = np.load(herb_emb_file)
    disease_embeddings = np.load(disease_emb_file)
    with id_mapping_file.open('rb') as handle:
        id_mapping = pickle.load(handle)
    herb_id_to_index = id_mapping['herb_id_to_index']
    disease_index_to_id = id_mapping['disease_index_to_id']

    print(f"  草药嵌入形状: {herb_embeddings.shape}")
    print(f"  疾病嵌入形状: {disease_embeddings.shape}")
    print(f"  草药数量: {len(herb_id_to_index)}")
    print(f"  疾病数量: {len(disease_index_to_id)}")

    target_key = str(target_herb_id)
    if target_key not in herb_id_to_index:
        print(f"错误: 目标草药 {target_herb_id} 不在 ID 映射中")
        return None
    target_herb_index = herb_id_to_index[target_key]
    target_herb_embedding = herb_embeddings[target_herb_index]

    print("\nStep 2: 计算全部疾病的关联得分...")
    logits = np.dot(disease_embeddings, target_herb_embedding)
    # Clipping is numerically equivalent at the reported precision and avoids
    # overflow for unusually large local embeddings.
    scores = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
    print(f"  得分范围: [{scores.min():.4f}, {scores.max():.4f}]")

    records = []
    for disease_idx, score in enumerate(scores):
        disease_id = int(disease_index_to_id.get(disease_idx, str(disease_idx)))
        records.append({'disease_id': disease_id, 'score': float(score)})
    results = pd.DataFrame(records).sort_values(
        'score', ascending=False
    ).reset_index(drop=True)
    results['rank'] = np.arange(1, len(results) + 1)

    print("\nStep 3: 加载 withheld associations...")
    removed_pairs_file = resolved_data_dir / f'{herb_name}_removed_pairs.txt'
    metrics = None
    if removed_pairs_file.is_file():
        true_associations = set()
        with removed_pairs_file.open('r', encoding='utf-8') as handle:
            for line in handle:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    true_associations.add(int(parts[1]))
        results['is_true_association'] = results['disease_id'].isin(true_associations)
        true_ranks = results.loc[results['is_true_association'], 'rank'].to_numpy()
        print(f"  withheld associations: {len(true_associations)}")
        if len(true_ranks):
            hits10 = int((true_ranks <= 10).sum())
            ideal10 = min(10, len(true_associations))
            dcg10 = sum(1.0 / np.log2(rank + 1) for rank in true_ranks if rank <= 10)
            idcg10 = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal10 + 1))
            metrics = {
                'herb': herb_name,
                'herb_id': target_herb_id,
                'known_associations': len(true_associations),
                'Hits@10': hits10,
                'Precision@10': hits10 / 10.0,
                'Recall@10': hits10 / len(true_associations),
                'MRR': 1.0 / float(true_ranks.min()),
                'NDCG@10': dcg10 / idcg10 if idcg10 else 0.0,
            }
    else:
        results['is_true_association'] = False
        print(f"  未找到 withheld-association 文件: {removed_pairs_file}")

    print("\nStep 4: 保存预测结果...")
    result_dir.mkdir(parents=True, exist_ok=True)
    output_file = result_dir / f'predictions_{herb_name}.csv'
    top100_file = result_dir / f'predictions_{herb_name}_top100.csv'
    results.to_csv(output_file, index=False)
    results.head(100).to_csv(top100_file, index=False)
    if metrics is not None:
        with (result_dir / 'ranking_metrics.json').open('w', encoding='utf-8') as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"  完整结果: {output_file}")
    print(f"  Top-100: {top100_file}")

    print("\nTop-10 预测结果:")
    for _, row in results.head(10).iterrows():
        marker = "✓" if row.get('is_true_association', False) else " "
        print(
            f"  {int(row['rank']):3d}. 疾病ID={int(row['disease_id']):5d}  "
            f"得分={row['score']:.4f}  [{marker}]"
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Rank diseases for a HerbHG case study')
    parser.add_argument(
        '--herb', required=True, choices=['ChuanXiong', 'HuangQi', 'both']
    )
    parser.add_argument(
        '--data-dir', help='包含 <Herb>_removed_pairs.txt 的案例数据目录。'
    )
    parser.add_argument(
        '--output-dir', help='包含 results_<Herb> 的案例结果根目录。'
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    herbs = list(HERB_IDS) if args.herb == 'both' else [args.herb]
    successful = True
    for herb_name in herbs:
        if predict_final_correct(
            herb_name,
            HERB_IDS[herb_name],
            result_root=args.output_dir,
            data_dir=args.data_dir,
        ) is None:
            successful = False
    return 0 if successful else 1


if __name__ == '__main__':
    raise SystemExit(main())
