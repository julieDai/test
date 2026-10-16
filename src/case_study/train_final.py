#!/usr/bin/env python3
# coding: utf-8
"""Train HerbHG for the withheld-association case study.

The command is independent of the caller's working directory. Relative
command-line paths are interpreted from the directory in which the command is
invoked; paths stored in a configuration are interpreted from that file.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import shlex
import sys
import time
import traceback
from pathlib import Path
from typing import Optional


CASE_STUDY_DIR = Path(__file__).resolve().parent
SOURCE_DIR = CASE_STUDY_DIR.parent
PROJECT_DIR = SOURCE_DIR.parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from herbhg.experiments.runtime import activate_tcm_backend

BACKEND_DIR = activate_tcm_backend(SOURCE_DIR)

from util.config import ModelConf
from util.paths import first_environment_value, resolve_path
from astra.runtime import configure_tcm_data_paths


HERB_IDS = {'ChuanXiong': 891, 'HuangQi': 2565}


def _load_herbhg_class():
    """Import the TensorFlow model only when training is requested."""

    from HerbHG import HerbHG

    return HerbHG


def _command_path(value: str | os.PathLike[str]) -> Path:
    """Resolve an explicit CLI path against the caller's working directory."""

    return resolve_path(value, Path.cwd())


def _environment_path(names, default: Path) -> Path:
    """Resolve a runtime path from the first configured environment variable."""

    value = first_environment_value(names)
    return resolve_path(value, PROJECT_DIR) if value else default.resolve(strict=False)


def _default_config(herb_name: str) -> Path:
    return CASE_STUDY_DIR / f'HerbHG_{herb_name}.conf'


def load_data_simple(data_file: str | os.PathLike[str]):
    """Load prepared three-column H--D examples without resampling."""

    data_path = Path(data_file).resolve(strict=False)
    print(f"加载数据文件: {data_path}")
    training_data = []
    with data_path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) < 3:
                raise ValueError(
                    f'{data_path}:{line_number} 需要至少三列 herb disease label'
                )
            training_data.append([parts[0], parts[1], float(parts[2])])

    print(f"数据加载完成: {len(training_data)} 条")
    pos_count = sum(1 for item in training_data if item[2] == 1.0)
    print(f"正样本: {pos_count}, 负样本: {len(training_data) - pos_count}")
    return training_data


def _configure_runtime(seed: int, device: str, output_dir: Path) -> None:
    """Set HerbHG runtime variables and compatibility aliases."""

    os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    os.environ['HERBHG_SEED'] = str(seed)
    os.environ['HERBHG_OUTPUT_DIR'] = str(output_dir)
    os.environ['HERBHG_LOG_DIR'] = str(output_dir / 'logs')
    os.environ['HERBHG_ALLOW_MISSING_SEMANTIC'] = '0'

    # Compatibility variables consumed by the retained numerical backend.
    os.environ['ASTRA_SEED'] = str(seed)
    os.environ['ASTRA_OUTPUT_DIR'] = str(output_dir)
    os.environ['ASTRA_LOG_DIR'] = str(output_dir / 'logs')
    os.environ['ASTRA_ALLOW_MISSING_SEMANTIC'] = '0'

    # Compatibility variables consumed by the retained backend modules.
    os.environ['CDRUG_SEED'] = str(seed)
    os.environ['CDRUG_OUTPUT_DIR'] = str(output_dir)
    os.environ['CDRUG_LOG_DIR'] = str(output_dir / 'logs')

    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)


def train_case_study_final(
    herb_name: str,
    target_herb_id: int,
    *,
    config_file: Optional[str | os.PathLike[str]] = None,
    data_dir: Optional[str | os.PathLike[str]] = None,
    source_data_dir: Optional[str | os.PathLike[str]] = None,
    result_root: Optional[str | os.PathLike[str]] = None,
    seed: int = 42,
    device: str = '0',
) -> bool:
    """Train one target-herb HerbHG model and save its entity embeddings."""

    print(f"\n{'=' * 80}")
    print(f"HerbHG case-study training - {herb_name} (ID={target_herb_id})")
    print(f"{'=' * 80}\n")

    if data_dir is None:
        resolved_data_dir = _environment_path(
            ('HERBHG_CASE_DATA_DIR', 'ASTRA_CASE_DATA_DIR', 'CDRUG_CASE_DATA_DIR'),
            PROJECT_DIR / 'data' / 'processed' / 'case_study',
        )
    else:
        resolved_data_dir = _command_path(data_dir)
    data_file = resolved_data_dir / f'H_D_{herb_name}.txt'
    if not data_file.is_file():
        print(f"错误: 数据文件不存在: {data_file}")
        print("请先运行 prepare_case_study_data.py")
        return False

    resolved_config = (
        _command_path(config_file) if config_file else _default_config(herb_name)
    )
    if not resolved_config.is_file():
        print(f"错误: 配置文件不存在: {resolved_config}")
        return False

    if result_root is None:
        resolved_result_root = _environment_path(
            ('HERBHG_CASE_RESULT_DIR', 'ASTRA_CASE_RESULT_DIR', 'CDRUG_CASE_RESULT_DIR'),
            PROJECT_DIR / 'outputs' / 'case_study',
        )
    else:
        resolved_result_root = _command_path(result_root)
    output_dir = resolved_result_root / f'results_{herb_name}'
    output_dir.mkdir(parents=True, exist_ok=True)
    _configure_runtime(seed, device, output_dir)

    if source_data_dir is None:
        resolved_source_data_dir = _environment_path(
            ('HERBHG_DATA_DIR', 'ASTRA_DATA_DIR', 'CDRUG_DATA_DIR'),
            PROJECT_DIR / 'data' / 'tcm_suite',
        )
    else:
        resolved_source_data_dir = _command_path(source_data_dir)

    try:
        import numpy as np

        conf = ModelConf(resolved_config)
        # Keep the prepared H--D examples separate while sourcing all graph
        # relations and biological features from one explicit TCM directory.
        configure_tcm_data_paths(conf, resolved_source_data_dir)
        conf.set('datapath', data_file)
        conf.set('model.name', 'HerbHG')
        conf.set('output.setup', f'on -dir {shlex.quote(str(output_dir))}')
        training_data = load_data_simple(data_file)

        print("\nInitializing HerbHG...")
        herbhg_class = _load_herbhg_class()
        import tensorflow.compat.v1 as tf

        tf.set_random_seed(seed)
        model = herbhg_class(conf, training_data, [], '[1]')
        model.readConfiguration()

        print("\n模型信息:")
        print(f"  草药数量: {model.num_herbs}")
        print(f"  疾病数量: {model.num_diseases}")
        print(f"  化学成分数量: {model.num_compounds}")
        print(f"  蛋白质数量: {model.num_proteins}")
        print(f"  训练样本数: {model.train_size}")
        print(f"  嵌入维度: {model.emb_size}")
        if hasattr(model, 'batch_size'):
            print(f"  Batch 大小: {model.batch_size}")
            expected_batches = (
                model.train_size + model.batch_size - 1
            ) // model.batch_size
            print(f"  预期 batch 数: {expected_batches}")
        if hasattr(model, 'maxEpoch'):
            print(f"  训练轮数: {model.maxEpoch}")

        print("\n初始化模型计算图...")
        model.initModel()
        print("  模型计算图初始化完成")

        print(f"\n{'=' * 80}\n开始训练...\n{'=' * 80}\n")
        start_time = time.time()
        model.trainModel()
        training_time = time.time() - start_time
        print(f"\n训练完成! 用时 {training_time:.2f} 秒 ({training_time / 60:.2f} 分钟)")

        herb_emb_file = output_dir / 'herb_embeddings.npy'
        disease_emb_file = output_dir / 'disease_embeddings.npy'
        np.save(herb_emb_file, model.u)
        np.save(disease_emb_file, model.i)
        print(f"  草药嵌入: {herb_emb_file}，形状: {model.u.shape}")
        print(f"  疾病嵌入: {disease_emb_file}，形状: {model.i.shape}")

        id_mapping = {
            'herb_id_to_index': model.data.herb,
            'disease_id_to_index': model.data.disease,
            'disease_index_to_id': {v: k for k, v in model.data.disease.items()},
        }
        mapping_file = output_dir / 'id_mapping.pkl'
        with mapping_file.open('wb') as handle:
            pickle.dump(id_mapping, handle)
        print(f"  ID 映射: {mapping_file}")
        print(f"\nHerbHG outputs: {output_dir}\n")
        return True
    except Exception as exc:
        print(f"\n训练过程中出现错误: {exc}")
        traceback.print_exc()
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Train a HerbHG withheld-association case study')
    parser.add_argument(
        '--herb', required=True, choices=['ChuanXiong', 'HuangQi', 'both']
    )
    parser.add_argument(
        '--config', help='单个草药使用的配置文件；相对路径按当前命令目录解析。'
    )
    parser.add_argument(
        '--data-dir', help='Directory containing H_D_<Herb>.txt.'
    )
    parser.add_argument(
        '--source-data-dir',
        help='TCM-suite directory containing graph relations and biological features.',
    )
    parser.add_argument(
        '--output-dir', help='Root directory for case-study results.'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=int(
            first_environment_value(('HERBHG_SEED', 'ASTRA_SEED', 'CDRUG_SEED'))
            or 42
        ),
    )
    parser.add_argument(
        '--device', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.herb == 'both' and args.config:
        parser.error('--config 只能与单个 --herb 一起使用')

    herbs = list(HERB_IDS) if args.herb == 'both' else [args.herb]
    outcomes = [
        train_case_study_final(
            herb,
            HERB_IDS[herb],
            config_file=args.config,
            data_dir=args.data_dir,
            source_data_dir=args.source_data_dir,
            result_root=args.output_dir,
            seed=args.seed,
            device=args.device,
        )
        for herb in herbs
    ]
    return 0 if all(outcomes) else 1


if __name__ == '__main__':
    raise SystemExit(main())
