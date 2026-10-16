"""Exactness checks for the scalable MilGNet encode/decode boundary."""

from __future__ import annotations

import types
import unittest

import numpy as np

from astra.official_baselines.milgnet_runner import (
    cyclic_balanced_epoch_positions,
    decode_official_model,
    make_meta_path_bags,
    make_meta_path_bags_vectorized,
    pack_neighbor_rows,
)


class MilGNetScalableTest(unittest.TestCase):
    def test_cyclic_blocks_are_balanced_deterministic_and_cover_rows(self):
        labels = np.asarray([1] * 10 + [0] * 10, dtype=np.float32)
        first_cycle = [
            cyclic_balanced_epoch_positions(labels, 4, seed=42, epoch=epoch)
            for epoch in range(5)
        ]
        second_cycle = [
            cyclic_balanced_epoch_positions(labels, 4, seed=42, epoch=epoch)
            for epoch in range(5, 10)
        ]
        for block in first_cycle + second_cycle:
            self.assertEqual(labels[block].sum(), 2)
            self.assertEqual(len(np.unique(block)), 4)
        self.assertEqual(
            first_cycle[0].tolist(),
            cyclic_balanced_epoch_positions(labels, 4, seed=42, epoch=0).tolist(),
        )
        self.assertEqual(set(np.concatenate(first_cycle).tolist()), set(range(20)))
        self.assertEqual(set(np.concatenate(second_cycle).tolist()), set(range(20)))
        first_positive_order = np.concatenate(first_cycle)[
            labels[np.concatenate(first_cycle)] == 1
        ]
        second_positive_order = np.concatenate(second_cycle)[
            labels[np.concatenate(second_cycle)] == 1
        ]
        self.assertNotEqual(first_positive_order.tolist(), second_positive_order.tolist())

    def test_vectorized_bags_match_released_order_padding_and_truncation(self):
        herb_rows = (
            np.asarray([1, 2], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.arange(30, dtype=np.int64),
        )
        disease_rows = (
            np.asarray([1], dtype=np.int64),
            np.asarray([0, 2, 3], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.arange(25, dtype=np.int64),
        )
        pairs = np.asarray([[0, 0], [0, 1], [1, 2], [2, 3]], dtype=np.int64)
        max_instances = 17
        expected = make_meta_path_bags(
            pairs,
            herb_rows,
            disease_rows,
            max_instances=max_instances,
            seed=42,
        )
        actual = make_meta_path_bags_vectorized(
            pairs,
            pack_neighbor_rows(herb_rows, max_instances),
            pack_neighbor_rows(disease_rows, max_instances),
            max_instances=max_instances,
            seed=42,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_fused_mean_decoder_matches_author_operations_and_gradients(self):
        import torch

        class MeanAggregator(torch.nn.Module):
            agg_type = "mean"

            def forward(self, hidden, bags):
                values = torch.cat(
                    [
                        hidden["drug"][bags[:, :, :2]],
                        hidden["disease"][bags[:, :, 2:]],
                    ],
                    dim=2,
                )
                return torch.mean(values, dim=2)

        class MIL(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.project = torch.nn.Sequential(
                    torch.nn.Linear(5, 5),
                    torch.nn.Tanh(),
                    torch.nn.Linear(5, 1, bias=False),
                )

            def forward(self, values):
                attention = torch.softmax(self.project(values), dim=1)
                return (values * attention).sum(dim=1), attention

        torch.manual_seed(42)
        model = types.SimpleNamespace(
            aggregator=MeanAggregator(),
            mil=True,
            mil_layer=MIL(),
            bag_predict=torch.nn.Linear(5, 1, bias=False),
            ins_predict=None,
        )
        bags = torch.randint(0, 4, (3, 7, 4), dtype=torch.long)
        reference_hidden = {
            "drug": torch.randn(4, 5, requires_grad=True),
            "disease": torch.randn(4, 5, requires_grad=True),
        }
        fused_hidden = {
            key: value.detach().clone().requires_grad_(True)
            for key, value in reference_hidden.items()
        }
        expected, _ = decode_official_model(
            model, reference_hidden, bags, fused_mean=False
        )
        expected.sum().backward()
        expected_parameter_gradients = [
            parameter.grad.detach().clone()
            for parameter in (*model.mil_layer.parameters(), *model.bag_predict.parameters())
        ]
        for parameter in (*model.mil_layer.parameters(), *model.bag_predict.parameters()):
            parameter.grad = None
        actual, _ = decode_official_model(model, fused_hidden, bags, fused_mean=True)
        actual.sum().backward()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for key in reference_hidden:
            torch.testing.assert_close(
                fused_hidden[key].grad,
                reference_hidden[key].grad,
                rtol=1e-5,
                atol=1e-6,
            )
        for actual_gradient, expected_gradient in zip(
            (parameter.grad for parameter in (*model.mil_layer.parameters(), *model.bag_predict.parameters())),
            expected_parameter_gradients,
        ):
            torch.testing.assert_close(
                actual_gradient, expected_gradient, rtol=1e-5, atol=1e-6
            )


if __name__ == "__main__":
    unittest.main()
