#!/usr/bin/env python3
"""Command-line entry point for the TCM-suite HerbHG implementation."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path


# Resolve imports from this file's location, never from the caller's CWD.
SOURCE_DIR = Path(__file__).resolve().parent
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from util.config import ModelConf, OptionConf
from util.paths import first_environment_value, resolve_config_path, resolve_path


DEFAULT_CONFIG = SOURCE_DIR / 'ASTRA.conf'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Train and evaluate HerbHG on the TCM-suite benchmark.'
    )
    parser.add_argument(
        '--config',
        default=str(DEFAULT_CONFIG),
        help='Model configuration file (default: bundled TCM-suite configuration).',
    )
    parser.add_argument(
        '--data-dir',
        help='Directory containing the TCM-suite relation and biological feature files.',
    )
    parser.add_argument(
        '--output-dir',
        help='Directory for predictions, metrics, embeddings, and logs.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=int(first_environment_value(('HERBHG_SEED', 'ASTRA_SEED', 'CDRUG_SEED')) or '42'),
        help='Random seed (default: HERBHG_SEED or 42).',
    )
    parser.add_argument(
        '--device',
        default=first_environment_value(('HERBHG_DEVICE', 'ASTRA_DEVICE', 'CUDA_VISIBLE_DEVICES')) or 'auto',
        help='Device policy: auto, cpu, cuda, or a CUDA device list such as 0 or 0,1.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print resolved paths and exit without loading data or TensorFlow.',
    )
    parser.add_argument(
        '--allow-missing-semantic',
        action='store_true',
        help=(
            'Allow the historical structure-only fallback when herb/disease '
            'semantic arrays are absent. The default requires both files.'
        ),
    )
    return parser


def configure_seed(seed: int) -> None:
    """Seed Python, NumPy, and TensorFlow when TensorFlow is installed."""

    os.environ['HERBHG_SEED'] = str(seed)
    os.environ['ASTRA_SEED'] = str(seed)
    os.environ['CDRUG_SEED'] = str(seed)
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow.compat.v1 as tf

        tf.set_random_seed(seed)
    except ImportError:
        # Lightweight inspection and dry-run usage do not require TensorFlow.
        pass


def configure_device(device: str) -> None:
    """Translate the public device option to TensorFlow's CUDA environment."""

    normalized = device.strip().lower()
    os.environ['HERBHG_DEVICE'] = device
    os.environ['ASTRA_DEVICE'] = device
    if normalized == 'auto':
        return
    if normalized == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif normalized == 'cuda':
        if os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '', '-1'):
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = device


def _load_runner():
    """Import the training orchestrator only when an actual run is requested."""

    from HDR import HDR

    return HDR


def _user_path(value: str) -> Path:
    """Resolve an explicit CLI path against the caller's working directory."""

    return resolve_path(value, Path.cwd())


def _apply_data_directory(config: ModelConf, directory: Path) -> None:
    files = {
        'datapath': ('H_D.txt', None),
        'hc.path': ('H_C.txt', ('ASTRA_HC_FILE', 'CDRUG_HC_FILE')),
        'cp.path': ('C_P.txt', ('ASTRA_CP_FILE', 'CDRUG_CP_FILE')),
        'pd.path': ('P_D.txt', ('ASTRA_PD_FILE', 'CDRUG_PD_FILE')),
        'node.hc.path': ('H_C.txt', ('ASTRA_NODE_HC_FILE', 'CDRUG_NODE_HC_FILE')),
        'node.pd.path': ('P_D.txt', ('ASTRA_NODE_PD_FILE', 'CDRUG_NODE_PD_FILE')),
        'negative.path': ('negativehdlist.txt', ('ASTRA_NEGATIVE_FILE', 'CDRUG_NEGATIVE_FILE')),
        'compound.embedding': (
            'compound_embedding_kpgt.txt',
            ('ASTRA_COMPOUND_EMBEDDINGS', 'CDRUG_COMPOUND_EMBEDDINGS'),
        ),
        'protein.embedding': (
            'protein_embedding_esm2.txt',
            ('ASTRA_PROTEIN_EMBEDDINGS', 'CDRUG_PROTEIN_EMBEDDINGS'),
        ),
    }
    config.set('data.dir', directory)
    for key, (filename, env_var) in files.items():
        path = directory / filename
        config.set(key, path)
        # An explicit CLI directory takes precedence over inherited legacy
        # environment variables.  The resolved paths remain in the config so
        # setting --data-dir does not accidentally opt into the special
        # "frozen negative file" behavior tied to CDRUG_NEGATIVE_FILE.
        for name in env_var or ():
            os.environ.pop(name, None)


def _configured_output_directory(config: ModelConf) -> Path:
    output = OptionConf(config['output.setup'])
    raw_path = output['-dir'] if output.contains('-dir') else '../outputs/tcm_suite'
    return config.resolve_path(raw_path)


def _require_semantic_inputs(config: ModelConf) -> tuple[Path, Path]:
    """Return semantic input paths or fail before importing TensorFlow."""

    herb_path = resolve_config_path(
        config,
        'herb.text.embedding',
        '../artifacts/tcm_suite/herb_semantic_embeddings_64d.npy',
        ('ASTRA_HERB_TEXT_EMBEDDINGS', 'CDRUG_HERB_TEXT_EMBEDDINGS'),
    )
    disease_path = resolve_config_path(
        config,
        'disease.text.embedding',
        '../artifacts/tcm_suite/disease_semantic_embeddings_64d.npy',
        ('ASTRA_DISEASE_TEXT_EMBEDDINGS', 'CDRUG_DISEASE_TEXT_EMBEDDINGS'),
    )
    missing = [path for path in (herb_path, disease_path) if not path.is_file()]
    if missing:
        formatted = '\n'.join(f'  - {path}' for path in missing)
        raise FileNotFoundError(
            'Required TCM-suite semantic inputs are missing:\n'
            f'{formatted}\n'
            'Generate/provide both arrays, or explicitly pass '
            '--allow-missing-semantic for the structure-only fallback.'
        )
    return herb_path, disease_path


def main(argv=None) -> int:
    """Run ASTRA and return a process-style status code."""

    args = build_parser().parse_args(argv)
    configure_device(str(args.device))

    config_path = _user_path(args.config)
    config = ModelConf(config_path)

    if args.data_dir:
        data_dir = _user_path(args.data_dir)
        _apply_data_directory(config, data_dir)
    else:
        data_dir = config.resolve_path(config['datapath']).parent
        config.set('data.dir', data_dir)

    if args.output_dir:
        output_dir = _user_path(args.output_dir)
    else:
        configured_environment_output = first_environment_value(
            ('HERBHG_OUTPUT_DIR', 'ASTRA_OUTPUT_DIR', 'CDRUG_OUTPUT_DIR')
        )
        if configured_environment_output:
            output_dir = resolve_path(configured_environment_output)
        else:
            output_dir = _configured_output_directory(config)

    os.environ['HERBHG_OUTPUT_DIR'] = str(output_dir)
    os.environ['ASTRA_OUTPUT_DIR'] = str(output_dir)
    os.environ['CDRUG_OUTPUT_DIR'] = str(output_dir)
    log_dir = first_environment_value(
        ('HERBHG_LOG_DIR', 'ASTRA_LOG_DIR', 'CDRUG_LOG_DIR')
    )
    if not log_dir or args.output_dir:
        log_dir = str(output_dir / 'logs')
    os.environ['HERBHG_LOG_DIR'] = log_dir
    os.environ['ASTRA_LOG_DIR'] = log_dir
    os.environ['CDRUG_LOG_DIR'] = log_dir

    print('Configuration:', config.path)
    print('Data directory:', data_dir)
    print('Output directory:', output_dir)
    print('Random seed:', args.seed)
    print('Device policy:', args.device)
    print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES', '<unchanged>'))

    if args.dry_run:
        return 0

    if args.allow_missing_semantic:
        os.environ['ASTRA_ALLOW_MISSING_SEMANTIC'] = '1'
    else:
        os.environ.pop('ASTRA_ALLOW_MISSING_SEMANTIC', None)
        try:
            _require_semantic_inputs(config)
        except FileNotFoundError as exc:
            print(str(exc), file=sys.stderr)
            return 2

    configure_seed(args.seed)

    started = time.time()
    _load_runner()(config).execute()
    print('Running time: %f s' % (time.time() - started))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
