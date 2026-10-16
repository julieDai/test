"""HerbHG component variants.

Variant implementations are imported lazily so configuration and path tools
remain usable without importing TensorFlow.  Existing imports such as
``from ablation_study import Variant1RandomInit`` continue to work.
"""

from importlib import import_module

__all__ = [
    'Variant1RandomInit',
    'Variant2NoText',
    'Variant3MeanPooling',
    'Variant4NoClique',
    'Variant5NoDiseaseHypergraph'
]

__version__ = '1.0.0'


_VARIANT_MODULES = {
    'Variant1RandomInit': 'variant_1_random_init',
    'Variant2NoText': 'variant_2_no_text',
    'Variant3MeanPooling': 'variant_3_mean_pooling',
    'Variant4NoClique': 'variant_4_no_clique',
    'Variant5NoDiseaseHypergraph': 'variant_5_no_disease_hypergraph',
}


def __getattr__(name):
    """Load a TensorFlow-backed variant only when it is requested."""

    try:
        module_name = _VARIANT_MODULES[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f'{__name__}.{module_name}'), name)
    globals()[name] = value
    return value
