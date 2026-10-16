# coding: utf-8
"""Semantic-adapter experiments built on the frozen ASTRARefined model.

``ASTRARefined.py`` is the code-bound implementation used by an already
completed outer-test result.  It must remain byte-for-byte available for that
result's provenance.  This module therefore adds the next adapter search as a
new model class while reusing the frozen graph, loss, sampling, checkpoint,
and training implementation through inheritance.

All new adapters consume 64-dimensional semantic inputs and return the same
64-dimensional latent width.  They use identity-preserving initialisation so
that their initial semantic representation is exactly the L2-normalised input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from ASTRARefined import ASTRARefined, ASTRARefinedOptions


SEMANTIC_ADAPTERS = frozenset(
    {
        "identity",
        "linear",
        "residual_linear",
        "gated_residual",
        "residual_mlp",
    }
)
EXPERIMENTAL_SEMANTIC_ADAPTERS = frozenset(
    {"residual_linear", "gated_residual", "residual_mlp"}
)
DEFAULT_MLP_HIDDEN_DIMENSION = 32


def _positive_int(value: object, *, key: str, maximum: int = 1024) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be an integer, got {value!r}") from error
    if parsed <= 0 or parsed > maximum:
        raise ValueError(f"{key} must be in 1..{maximum}, got {parsed}")
    return parsed


@dataclass(frozen=True)
class ASTRASemanticRefinedOptions(ASTRARefinedOptions):
    """Frozen ASTRA controls plus the isolated semantic-adapter controls."""

    semantic_mlp_hidden_dim: int = DEFAULT_MLP_HIDDEN_DIMENSION

    @classmethod
    def from_mapping(
        cls, config: Mapping[str, object]
    ) -> "ASTRASemanticRefinedOptions":
        adapter = str(
            config.get("astra.semantic.adapter", "identity")
        ).strip().lower()
        if adapter not in SEMANTIC_ADAPTERS:
            raise ValueError(
                "astra.semantic.adapter must be one of "
                + ", ".join(sorted(SEMANTIC_ADAPTERS))
            )
        try:
            input_dimension = int(
                str(config.get("astra.semantic.input_dim", 64)).strip()
            )
        except (TypeError, ValueError) as error:
            raise ValueError(
                "astra.semantic.input_dim must be an integer"
            ) from error
        if adapter in EXPERIMENTAL_SEMANTIC_ADAPTERS and input_dimension != 64:
            raise ValueError(
                f"{adapter} requires astra.semantic.input_dim=64"
            )

        # Let the frozen parser continue to own every pre-existing control.
        # It does not know the three new enum values, so present the
        # behaviourally compatible identity value only while it parses.
        compatible = dict(config)
        if adapter in EXPERIMENTAL_SEMANTIC_ADAPTERS:
            compatible["astra.semantic.adapter"] = "identity"
        base = ASTRARefinedOptions.from_mapping(compatible)
        hidden = _positive_int(
            config.get(
                "astra.semantic.mlp_hidden_dim",
                DEFAULT_MLP_HIDDEN_DIMENSION,
            ),
            key="astra.semantic.mlp_hidden_dim",
        )
        return cls(
            **{
                **base.__dict__,
                "semantic_adapter": adapter,
                "semantic_mlp_hidden_dim": hidden,
            }
        )

    def as_jsonable(self) -> dict[str, object]:
        payload = super().as_jsonable()
        payload["semantic_mlp_hidden_dim"] = self.semantic_mlp_hidden_dim
        return payload


class ASTRASemanticRefined(ASTRARefined):
    """ASTRARefined with an isolated, identity-initialised adapter family."""

    def __init__(self, conf, trainingSet=None, testSet=None, fold="[1]"):
        requested_options = ASTRASemanticRefinedOptions.from_mapping(conf.config)
        requested_adapter = requested_options.semantic_adapter

        # ASTRARefined.__init__ only parses options after constructing the base
        # data object; model graph construction happens later in initModel.
        # Temporarily exposing ``identity`` lets us reuse that constructor
        # without modifying the frozen source file.  The original ModelConf is
        # restored even if construction raises.
        sentinel = object()
        original = conf.config.get("astra.semantic.adapter", sentinel)
        if requested_adapter in EXPERIMENTAL_SEMANTIC_ADAPTERS:
            conf.config["astra.semantic.adapter"] = "identity"
        try:
            super().__init__(conf, trainingSet, testSet, fold)
        finally:
            if original is sentinel:
                conf.config.pop("astra.semantic.adapter", None)
            else:
                conf.config["astra.semantic.adapter"] = original

        self.options = requested_options
        self.resolved_hyperparameters.update(
            {
                "model_class": "ASTRASemanticRefined",
                "c0_compatible": requested_options.c0_compatible,
                **requested_options.as_jsonable(),
            }
        )

    def _residual_linear(self, inputs, *, entity: str):
        width = self.emb_size
        zero = tf.zeros_initializer()
        matrix = self._weight(
            f"semantic_residual_projection_{entity}",
            (width, width),
            f"semantic_residual_projection_{entity}",
            zero,
        )
        bias = self._weight(
            f"semantic_residual_bias_{entity}",
            (1, width),
            f"semantic_residual_bias_{entity}",
            zero,
        )
        return inputs + tf.matmul(inputs, matrix) + bias

    def _gated_residual(self, inputs, *, entity: str):
        width = self.emb_size
        zero = tf.zeros_initializer()
        matrix = self._weight(
            f"semantic_gated_projection_{entity}",
            (width, width),
            f"semantic_gated_projection_{entity}",
            zero,
        )
        bias = self._weight(
            f"semantic_gated_bias_{entity}",
            (1, width),
            f"semantic_gated_bias_{entity}",
            zero,
        )
        gate_logit = self._weight(
            f"semantic_gate_logit_{entity}",
            (1, width),
            f"semantic_gate_logit_{entity}",
            zero,
        )
        update = tf.matmul(inputs, matrix) + bias
        return inputs + tf.nn.sigmoid(gate_logit) * update

    def _residual_mlp(self, inputs, *, entity: str, initializer):
        width = self.emb_size
        hidden_width = self.options.semantic_mlp_hidden_dim
        zero = tf.zeros_initializer()
        input_matrix = self._weight(
            f"semantic_mlp_input_{entity}",
            (width, hidden_width),
            f"semantic_mlp_input_{entity}",
            initializer,
        )
        input_bias = self._weight(
            f"semantic_mlp_input_bias_{entity}",
            (1, hidden_width),
            f"semantic_mlp_input_bias_{entity}",
            zero,
        )
        output_matrix = self._weight(
            f"semantic_mlp_output_{entity}",
            (hidden_width, width),
            f"semantic_mlp_output_{entity}",
            zero,
        )
        output_bias = self._weight(
            f"semantic_mlp_output_bias_{entity}",
            (1, width),
            f"semantic_mlp_output_bias_{entity}",
            zero,
        )
        hidden = tf.nn.leaky_relu(
            tf.matmul(inputs, input_matrix) + input_bias
        )
        return inputs + tf.matmul(hidden, output_matrix) + output_bias

    def _prepare_semantic_embeddings(self, initializer):
        adapter = self.options.semantic_adapter
        if adapter not in EXPERIMENTAL_SEMANTIC_ADAPTERS:
            return super()._prepare_semantic_embeddings(initializer)
        if not self.use_text_embeddings:
            return None, None

        herb_dim = int(self.herb_text_embeddings.shape.as_list()[1])
        disease_dim = int(self.disease_text_embeddings.shape.as_list()[1])
        expected = self.options.semantic_input_dim
        if herb_dim != expected or disease_dim != expected:
            raise ValueError(
                "semantic matrix width differs from astra.semantic.input_dim: "
                f"herb={herb_dim}, disease={disease_dim}, configured={expected}"
            )
        if expected != self.emb_size:
            raise ValueError(
                f"{adapter} requires semantic width to equal num.factors "
                f"({self.emb_size})"
            )

        herb_input = self.herb_text_embeddings
        disease_input = self.disease_text_embeddings
        if self.options.semantic_freeze_input:
            herb_input = tf.stop_gradient(herb_input)
            disease_input = tf.stop_gradient(disease_input)

        if adapter == "residual_linear":
            herb = self._residual_linear(herb_input, entity="h")
            disease = self._residual_linear(disease_input, entity="d")
        elif adapter == "gated_residual":
            herb = self._gated_residual(herb_input, entity="h")
            disease = self._gated_residual(disease_input, entity="d")
        else:
            herb = self._residual_mlp(
                herb_input, entity="h", initializer=initializer
            )
            disease = self._residual_mlp(
                disease_input, entity="d", initializer=initializer
            )

        return (
            tf.nn.l2_normalize(herb, axis=1),
            tf.nn.l2_normalize(disease, axis=1),
        )


def semantic_adapter_parameter_count(
    adapter: str,
    *,
    width: int = 64,
    hidden_width: int = DEFAULT_MLP_HIDDEN_DIMENSION,
    entity_count: int = 2,
) -> int:
    """Return trainable adapter parameters, excluding semantic input tensors."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (width, hidden_width, entity_count)
    ):
        raise ValueError("adapter dimensions and entity_count must be positive integers")
    normalized = str(adapter).strip().lower()
    if normalized == "identity":
        per_entity = 0
    elif normalized in {"linear", "residual_linear"}:
        per_entity = width * width + width
    elif normalized == "gated_residual":
        per_entity = width * width + width + width
    elif normalized == "residual_mlp":
        per_entity = (
            width * hidden_width
            + hidden_width
            + hidden_width * width
            + width
        )
    else:
        raise ValueError(f"unsupported semantic adapter: {adapter!r}")
    result = entity_count * per_entity
    if not math.isfinite(float(result)):
        raise ValueError("adapter parameter count overflowed")
    return result
