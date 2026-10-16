# coding: utf-8
"""TCM-only residual semantic adapter without modifying the frozen HGHDA.

The strict TCM implementation loads the audited 64-dimensional herb and
disease matrices in ``base.herbRecommender`` and first consumes them inside
``HGHDA.initModel``.  This subclass intercepts only that consumption point.
All historical structure propagation, scoring, loss, batching, and optimiser
code is otherwise retained.

The residual MLP is identity-initialised.  Its output projection and output
bias start at zero, so the first forward pass is exactly the identity control.
Adapter variables are created only after HGHDA has created every shared model
weight.  Consequently the adapter arm cannot perturb the seeded initialisation
stream of shared variables.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from time import localtime, strftime, time
from typing import Mapping

import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from HGHDA import HGHDA


ADAPTERS = frozenset({"identity", "residual_mlp"})
SEMANTIC_WIDTH = 64
MLP_HIDDEN_WIDTH = 32
ADAPTER_WEIGHT_PREFIX = "semantic_adapter_"
AUDIT_SCHEMA = "astra-tcm-semantic-adapter-audit/v1"


def _parse_bool(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{key} must be a boolean")


def _positive_int(value: object, *, key: str) -> int:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be an integer") from error
    if parsed <= 0:
        raise ValueError(f"{key} must be positive")
    return parsed


@dataclass(frozen=True)
class TCMAdapterOptions:
    adapter: str = "identity"
    input_dimension: int = SEMANTIC_WIDTH
    hidden_dimension: int = MLP_HIDDEN_WIDTH
    freeze_input: bool = False

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> "TCMAdapterOptions":
        adapter = str(config.get("astra.semantic.adapter", "identity")).strip().lower()
        if adapter not in ADAPTERS:
            raise ValueError(
                "astra.semantic.adapter must be identity or residual_mlp"
            )
        input_dimension = _positive_int(
            config.get("astra.semantic.input_dim", SEMANTIC_WIDTH),
            key="astra.semantic.input_dim",
        )
        hidden_dimension = _positive_int(
            config.get("astra.semantic.mlp_hidden_dim", MLP_HIDDEN_WIDTH),
            key="astra.semantic.mlp_hidden_dim",
        )
        freeze_input = _parse_bool(
            config.get("astra.semantic.freeze_input", False),
            key="astra.semantic.freeze_input",
        )
        if input_dimension != SEMANTIC_WIDTH:
            raise ValueError("TCM semantic-adapter stage fixes input_dim=64")
        if hidden_dimension != MLP_HIDDEN_WIDTH:
            raise ValueError("TCM semantic-adapter stage fixes mlp_hidden_dim=32")
        if freeze_input:
            raise ValueError("TCM semantic-adapter stage fixes freeze_input=false")
        return cls(
            adapter=adapter,
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            freeze_input=freeze_input,
        )

    def as_jsonable(self) -> dict[str, object]:
        return {
            "semantic_adapter": self.adapter,
            "semantic_input_dim": self.input_dimension,
            "semantic_mlp_hidden_dim": self.hidden_dimension,
            "semantic_freeze_input": self.freeze_input,
        }


def semantic_adapter_parameter_count(
    adapter: str,
    *,
    width: int = SEMANTIC_WIDTH,
    hidden_width: int = MLP_HIDDEN_WIDTH,
    entity_count: int = 2,
) -> int:
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (width, hidden_width, entity_count)
    ):
        raise ValueError("adapter dimensions and entity_count must be positive integers")
    normalized = str(adapter).strip().lower()
    if normalized == "identity":
        return 0
    if normalized != "residual_mlp":
        raise ValueError(f"unsupported semantic adapter: {adapter!r}")
    per_entity = width * hidden_width + hidden_width + hidden_width * width + width
    result = entity_count * per_entity
    if not math.isfinite(float(result)):
        raise ValueError("adapter parameter count overflowed")
    return result


def _array_sha256(values: object) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class ASTRATCMSemanticAdapter(HGHDA):
    """HGHDA with a controlled adapter on the audited 64D TCM semantics."""

    def __init__(self, conf, trainingSet=None, testSet=None, fold="[1]"):
        self.adapter_options = TCMAdapterOptions.from_mapping(conf.config)
        alpha = float(conf.config.get("text.fusion.alpha", "1.0"))
        if not math.isclose(alpha, 1.0, rel_tol=0.0, abs_tol=1e-15):
            raise ValueError("TCM semantic-adapter stage fixes text.fusion.alpha=1")
        self._semantic_forward_building = False
        self._semantic_adapter_built = False
        self._adapter_weight_keys: tuple[str, ...] = ()
        super(ASTRATCMSemanticAdapter, self).__init__(
            conf,
            trainingSet=trainingSet,
            testSet=testSet,
            fold=fold,
            text_fusion_alpha=alpha,
        )
        self.resolved_adapter_hyperparameters = {
            "model_class": type(self).__name__,
            "text_fusion_alpha": alpha,
            **self.adapter_options.as_jsonable(),
        }

    @property
    def herb_text_embeddings(self):
        raw = getattr(self, "_raw_herb_text_embeddings", None)
        if raw is None:
            raise AttributeError("herb_text_embeddings has not been loaded")
        if self._semantic_forward_building:
            self._ensure_semantic_adapter()
            return self._adapted_herb_text_embeddings
        return raw

    @herb_text_embeddings.setter
    def herb_text_embeddings(self, value) -> None:
        self._raw_herb_text_embeddings = value

    @property
    def disease_text_embeddings(self):
        raw = getattr(self, "_raw_disease_text_embeddings", None)
        if raw is None:
            raise AttributeError("disease_text_embeddings has not been loaded")
        if self._semantic_forward_building:
            self._ensure_semantic_adapter()
            return self._adapted_disease_text_embeddings
        return raw

    @disease_text_embeddings.setter
    def disease_text_embeddings(self, value) -> None:
        self._raw_disease_text_embeddings = value

    def _adapter_weight(self, key: str, shape: tuple[int, int], initializer):
        variable = tf.Variable(initializer(shape), name=key)
        self.weights[key] = variable
        return variable

    def _residual_mlp(self, inputs, *, entity: str, initializer):
        width = int(self.emb_size)
        hidden = self.adapter_options.hidden_dimension
        zero = tf.zeros_initializer()
        input_matrix = self._adapter_weight(
            f"{ADAPTER_WEIGHT_PREFIX}mlp_input_{entity}",
            (width, hidden),
            initializer,
        )
        input_bias = self._adapter_weight(
            f"{ADAPTER_WEIGHT_PREFIX}mlp_input_bias_{entity}",
            (1, hidden),
            zero,
        )
        output_matrix = self._adapter_weight(
            f"{ADAPTER_WEIGHT_PREFIX}mlp_output_{entity}",
            (hidden, width),
            zero,
        )
        output_bias = self._adapter_weight(
            f"{ADAPTER_WEIGHT_PREFIX}mlp_output_bias_{entity}",
            (1, width),
            zero,
        )
        hidden_values = tf.nn.leaky_relu(
            tf.matmul(inputs, input_matrix) + input_bias
        )
        return inputs + tf.matmul(hidden_values, output_matrix) + output_bias

    def _ensure_semantic_adapter(self) -> None:
        if self._semantic_adapter_built:
            return
        if not hasattr(self, "weights"):
            raise RuntimeError(
                "semantic adapter was requested before HGHDA created shared weights"
            )
        herb = self._raw_herb_text_embeddings
        disease = self._raw_disease_text_embeddings
        herb_width = int(herb.shape.as_list()[1])
        disease_width = int(disease.shape.as_list()[1])
        if (
            herb_width != self.adapter_options.input_dimension
            or disease_width != self.adapter_options.input_dimension
            or int(self.emb_size) != self.adapter_options.input_dimension
        ):
            raise ValueError(
                "TCM adapter requires herb, disease, and latent widths all equal 64"
            )

        before = tuple(self.weights)
        if self.adapter_options.adapter == "identity":
            adapted_herb = herb
            adapted_disease = disease
        else:
            initializer = tf.keras.initializers.glorot_normal()
            adapted_herb = self._residual_mlp(
                herb, entity="herb", initializer=initializer
            )
            adapted_disease = self._residual_mlp(
                disease, entity="disease", initializer=initializer
            )
        self._adapter_weight_keys = tuple(
            key for key in self.weights if key not in before
        )
        self._adapted_herb_text_embeddings = adapted_herb
        self._adapted_disease_text_embeddings = adapted_disease
        self._semantic_adapter_built = True

        observed = sum(
            int(np.prod(self.weights[key].shape.as_list()))
            for key in self._adapter_weight_keys
        )
        expected = semantic_adapter_parameter_count(self.adapter_options.adapter)
        if observed != expected:
            raise RuntimeError(
                f"semantic adapter parameter count is {observed}, expected {expected}"
            )

    @staticmethod
    def _gradient_norm(gradient):
        if gradient is None:
            return None
        return tf.global_norm([gradient])

    def _build_adapter_audit_graph(self) -> None:
        probe = tf.reduce_sum(self.final_hcedge[:1]) + tf.reduce_sum(
            self.final_pdedge[:1]
        )
        adapter_variables = [self.weights[key] for key in self._adapter_weight_keys]
        adapter_gradients = tf.gradients(probe, adapter_variables)
        raw_variables = [
            self._raw_herb_text_embeddings,
            self._raw_disease_text_embeddings,
        ]
        raw_gradients = tf.gradients(probe, raw_variables)
        self._adapter_forward_connected = {
            key: gradient is not None
            for key, gradient in zip(self._adapter_weight_keys, adapter_gradients)
        }
        self._adapter_gradient_norm_tensors = {
            key: self._gradient_norm(gradient)
            for key, gradient in zip(self._adapter_weight_keys, adapter_gradients)
            if gradient is not None
        }
        self._raw_semantic_forward_connected = {
            entity: gradient is not None
            for entity, gradient in zip(("herb", "disease"), raw_gradients)
        }
        self._raw_gradient_norm_tensors = {
            entity: self._gradient_norm(gradient)
            for entity, gradient in zip(("herb", "disease"), raw_gradients)
            if gradient is not None
        }

    def initModel(self):
        self._semantic_forward_building = True
        try:
            super(ASTRATCMSemanticAdapter, self).initModel()
        finally:
            # The strict feature binder must continue seeing the raw audited
            # variables and their ``initial_value`` tensors after graph build.
            self._semantic_forward_building = False
        if not self._semantic_adapter_built:
            raise RuntimeError("HGHDA did not consume the semantic adapter")
        self._build_adapter_audit_graph()

    def _variable_hashes(self, variables: Mapping[str, object]) -> dict[str, str]:
        if not variables:
            return {}
        values = self.sess.run(
            dict(variables), feed_dict={self.isTraining: 0}
        )
        return {key: _array_sha256(value) for key, value in values.items()}

    def _gradient_norms(self, tensors: Mapping[str, object]) -> dict[str, float]:
        if not tensors:
            return {}
        values = self.sess.run(dict(tensors), feed_dict={self.isTraining: 0})
        return {key: float(value) for key, value in values.items()}

    def _semantic_delta(self) -> dict[str, float]:
        raw_herb, raw_disease, adapted_herb, adapted_disease = self.sess.run(
            [
                self._raw_herb_text_embeddings,
                self._raw_disease_text_embeddings,
                self._adapted_herb_text_embeddings,
                self._adapted_disease_text_embeddings,
            ],
            feed_dict={self.isTraining: 0},
        )
        return {
            "herb_max_abs": float(np.max(np.abs(adapted_herb - raw_herb))),
            "disease_max_abs": float(
                np.max(np.abs(adapted_disease - raw_disease))
            ),
        }

    def _shared_variables(self) -> dict[str, object]:
        adapter_keys = set(self._adapter_weight_keys)
        shared = {
            f"weight/{key}": value
            for key, value in self.weights.items()
            if key not in adapter_keys
        }
        shared.update(
            {
                "input/herb": self._raw_herb_text_embeddings,
                "input/disease": self._raw_disease_text_embeddings,
                "input/compound": self.compound_embeddings,
                "input/protein": self.protein_embeddings,
            }
        )
        return shared

    def _adapter_variables(self) -> dict[str, object]:
        return {key: self.weights[key] for key in self._adapter_weight_keys}

    def _audit_snapshot(self) -> dict[str, object]:
        return {
            "shared_variable_sha256": self._variable_hashes(
                self._shared_variables()
            ),
            "adapter_variable_sha256": self._variable_hashes(
                self._adapter_variables()
            ),
            "semantic_delta": self._semantic_delta(),
            "adapter_forward_gradient_norm": self._gradient_norms(
                self._adapter_gradient_norm_tensors
            ),
            "raw_semantic_forward_gradient_norm": self._gradient_norms(
                self._raw_gradient_norm_tensors
            ),
        }

    def _write_adapter_audit(
        self,
        *,
        output_dir: Path,
        initial: Mapping[str, object],
        final: Mapping[str, object],
        optimizer_steps: int,
    ) -> Path:
        adapter_variables = self._adapter_variables()
        trainable = tuple(tf.trainable_variables())
        raw_trainable = {
            "herb": any(
                value is self._raw_herb_text_embeddings for value in trainable
            ),
            "disease": any(
                value is self._raw_disease_text_embeddings for value in trainable
            ),
        }
        initial_adapter_hashes = dict(initial["adapter_variable_sha256"])
        final_adapter_hashes = dict(final["adapter_variable_sha256"])
        receipt = {
            "schema": AUDIT_SCHEMA,
            "adapter": self.adapter_options.adapter,
            "options": self.adapter_options.as_jsonable(),
            "adapter_parameter_count": semantic_adapter_parameter_count(
                self.adapter_options.adapter
            ),
            "adapter_variables": {
                key: {
                    "name": variable.name,
                    "shape": variable.shape.as_list(),
                    "trainable": any(value is variable for value in trainable),
                }
                for key, variable in adapter_variables.items()
            },
            "raw_semantic_trainable": raw_trainable,
            "forward_dependency": {
                "adapter": self._adapter_forward_connected,
                "raw_semantics": self._raw_semantic_forward_connected,
                "probe_excludes_regularization": True,
            },
            "initial": dict(initial),
            "final": dict(final),
            "adapter_parameters_changed": initial_adapter_hashes
            != final_adapter_hashes,
            "optimizer_steps": int(optimizer_steps),
            "model_source": {
                "path": str(Path(__file__).resolve()),
                "sha256": _file_sha256(Path(__file__).resolve()),
            },
        }
        path = output_dir / "tcm_semantic_adapter_audit.json"
        _atomic_json(path, receipt)
        self.adapter_audit_receipt = receipt
        self.adapter_audit_receipt_path = path
        return path

    def trainModel(self):
        # This is the frozen HGHDA objective and optimiser with two additions:
        # exact pre/post adapter audits and an optimiser-step counter.
        sigmoid_output = tf.sigmoid(
            tf.reduce_sum(tf.multiply(self.u_embedding, self.v_embedding), 1)
        )
        sigmoid_output = tf.clip_by_value(sigmoid_output, 1e-7, 1.0 - 1e-7)
        y = tf.reduce_sum(
            tf.multiply(self.neg_disease_embedding, tf.log(sigmoid_output))
            + tf.multiply(
                (1 - self.neg_disease_embedding), tf.log(1 - sigmoid_output)
            ),
            0,
        )
        reg_loss = 0
        for key in self.weights:
            reg_loss += 0.001 * tf.nn.l2_loss(self.weights[key])
        reg_loss += self.regU * (
            tf.nn.l2_loss(self.final_hcedge)
            + tf.nn.l2_loss(self.final_pdedge)
        )
        loss = -tf.reduce_sum(y) + reg_loss
        optimizer = tf.train.AdamOptimizer(self.lRate)
        train = optimizer.minimize(loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

        initial = self._audit_snapshot()
        initial_delta = initial["semantic_delta"]
        if any(float(value) != 0.0 for value in initial_delta.values()):
            raise RuntimeError(
                "semantic adapter is not identity-preserving at initialization"
            )
        optimizer_steps = 0
        for epoch in range(self.maxEpoch):
            for batch_index, batch in enumerate(self.next_batch_pairwise()):
                herb_idx, disease_idx, labels = batch
                _, loss_value = self.sess.run(
                    [train, loss],
                    feed_dict={
                        self.u_idx: herb_idx,
                        self.neg_idx: labels,
                        self.v_idx: disease_idx,
                        self.isTraining: 1,
                    },
                )
                optimizer_steps += 1
                if batch_index % 100 == 0:
                    print(
                        "training:",
                        epoch + 1,
                        "batch",
                        batch_index,
                        "loss:",
                        loss_value,
                        flush=True,
                    )

        self.u, self.i, self.weight = self.sess.run(
            [self.final_hcedge, self.final_pdedge, self.weights],
            feed_dict={self.isTraining: 0},
        )
        final = self._audit_snapshot()
        output_dir = Path(
            os.environ.get(
                "ASTRA_OUTPUT_DIR",
                os.environ.get("CDRUG_OUTPUT_DIR", "./results"),
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        current_time = strftime("%Y-%m-%d %H-%M-%S", localtime(time()))
        np.savetxt(
            output_dir / ("herbedgeDHCN_herb_embedding" + current_time + ".txt"),
            self.u,
        )
        np.savetxt(
            output_dir
            / ("diseaseedgeDHCN_disease_embedding" + current_time + ".txt"),
            self.i,
        )
        self._write_adapter_audit(
            output_dir=output_dir,
            initial=initial,
            final=final,
            optimizer_steps=optimizer_steps,
        )
