"""CPU-only invariant tests using synthetic tensors; no video data or model."""

import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import unittest
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from ssl_primitives import make_mask, masked_patch_mse, patchify, unpatchify


class PrimitiveTests(unittest.TestCase):
    def test_roundtrip_and_known_patch_coordinates(self):
        c = torch.arange(3).view(1, 3, 1, 1, 1) * 10000
        t = torch.arange(4).view(1, 1, 4, 1, 1) * 100
        h = torch.arange(8).view(1, 1, 1, 8, 1) * 10
        w = torch.arange(8).view(1, 1, 1, 1, 8)
        x = c + t + h + w
        patches = patchify(x)
        self.assertEqual(tuple(patches.shape), (1, 2, 2, 2, 96))
        expected = torch.tensor([
            channel * 10000 + frame * 100 + row * 10 + col
            for frame in (2, 3)
            for row in range(4, 8)
            for col in range(4, 8)
            for channel in range(3)
        ])
        self.assertTrue(torch.equal(patches[0, 1, 1, 1], expected))
        self.assertTrue(torch.equal(unpatchify(patches), x))
        noncontiguous = x.transpose(-1, -2)
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertTrue(torch.equal(unpatchify(patchify(noncontiguous)), noncontiguous))
        x_float = x.to(torch.float64).repeat(2, 1, 1, 1, 1)
        self.assertTrue(torch.equal(unpatchify(patchify(x_float, (1, 2, 2)), (1, 2, 2)), x_float))

    def test_patchify_autograd_roundtrip(self):
        x = torch.arange(3 * 4 * 8 * 8, dtype=torch.float64).reshape(1, 3, 4, 8, 8)
        x.requires_grad_()
        unpatchify(patchify(x)).sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.ones_like(x)))

    def test_mask_modes_exact_count_and_seed(self):
        grid = (4, 4, 6)
        for mode in ("spatial_tube", "temporal_block", "random_3d"):
            with self.subTest(mode=mode):
                a = make_mask(3, grid, 0.5, mode, torch.Generator().manual_seed(20261026))
                b = make_mask(3, grid, 0.5, mode, torch.Generator().manual_seed(20261026))
                self.assertEqual(a.dtype, torch.bool)
                self.assertEqual(a.device.type, "cpu")
                self.assertEqual(tuple(a.shape), (3, 4, 4, 6))
                self.assertTrue(torch.equal(a, b))
                self.assertTrue(torch.equal(a.flatten(1).sum(1), torch.tensor([48, 48, 48])))
                if mode == "spatial_tube":
                    self.assertTrue(torch.equal(a, a[:, :1].expand_as(a)))
                    self.assertTrue(torch.equal(a[:, 0].flatten(1).sum(1), torch.tensor([12, 12, 12])))
                if mode == "temporal_block":
                    for sample in a:
                        masked_slices = sample.flatten(1).all(1)
                        self.assertTrue(torch.equal(sample, masked_slices[:, None, None].expand_as(sample)))
                        positions = torch.nonzero(masked_slices).flatten()
                        self.assertEqual(positions.numel(), 2)
                        self.assertEqual(int(positions[1] - positions[0]), 1)

    def test_mask_boundary_ratios_and_no_global_rng_side_effect(self):
        for mode in ("spatial_tube", "temporal_block", "random_3d"):
            self.assertFalse(bool(make_mask(2, (4, 4, 6), 0, mode).any()))
            self.assertTrue(bool(make_mask(2, (4, 4, 6), 1, mode).all()))
        before = torch.random.get_rng_state().clone()
        make_mask(2, (4, 4, 6), 0.5, "spatial_tube", torch.Generator().manual_seed(7))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        fractional_decimal = make_mask(1, (2, 5, 10), 0.58, "spatial_tube")
        self.assertEqual(int(fractional_decimal.sum()), 58)

    def test_mask_and_patch_input_errors(self):
        invalid_calls = [
            lambda: patchify(torch.zeros(1, 1, 4, 8, 8)),
            lambda: patchify(torch.zeros(1, 3, 3, 8, 8)),
            lambda: patchify(torch.zeros(1, 3, 4, 7, 8)),
            lambda: patchify(torch.zeros(1, 3, 4, 8, 8), (0, 4, 4)),
            lambda: patchify(torch.zeros(1, 3, 4, 8, 8), (2, 4.0, 4)),
            lambda: patchify(torch.zeros(0, 3, 4, 8, 8)),
            lambda: unpatchify(torch.zeros(1, 2, 2, 2, 95)),
            lambda: make_mask(0, (4, 4, 4), 0.5, "spatial_tube"),
            lambda: make_mask(True, (4, 4, 4), 0.5, "spatial_tube"),
            lambda: make_mask(1, (4, 0, 4), 0.5, "spatial_tube"),
            lambda: make_mask(1, (4, 4), 0.5, "spatial_tube"),
            lambda: make_mask(1, (4, 4, 4), float("nan"), "spatial_tube"),
            lambda: make_mask(1, (4, 4, 4), float("inf"), "spatial_tube"),
            lambda: make_mask(1, (4, 4, 4), -0.1, "spatial_tube"),
            lambda: make_mask(1, (4, 4, 4), 1.1, "spatial_tube"),
            lambda: make_mask(1, (4, 4, 4), True, "spatial_tube"),
            lambda: make_mask(1, (3, 3, 3), 0.5, "random_3d"),
            lambda: make_mask(1, (2, 3, 3), 0.5, "spatial_tube"),
            lambda: make_mask(1, (3, 4, 4), 0.5, "temporal_block"),
            lambda: make_mask(1, (4, 4, 4), 0.5, "ambiguous_tube"),
        ]
        for number, call in enumerate(invalid_calls):
            with self.subTest(case=number):
                with self.assertRaises(ValueError):
                    call()
        with self.assertRaises(TypeError):
            patchify([1, 2])
        with self.assertRaises(TypeError):
            make_mask(1, (4, 4, 4), 0.5, "spatial_tube", 7)

    def test_masked_loss_value_invariance_and_gradient(self):
        pred = torch.tensor([1., 2., 3., 4., 5., 6., 7., 8.], dtype=torch.float64)
        pred = pred.reshape(1, 1, 2, 2, 2).requires_grad_()
        target = torch.zeros_like(pred, requires_grad=True)
        mask = torch.tensor([[[[True, False], [True, False]]]])
        loss = masked_patch_mse(pred, target, mask)
        # Selected vectors [1,2] and [5,6]: (1+4+25+36)/4.
        self.assertEqual(float(loss), 16.5)
        changed_target = target.detach().clone()
        changed_target[~mask] += 1000
        self.assertEqual(float(masked_patch_mse(pred, changed_target, mask)), 16.5)
        changed_pred = pred.detach().clone()
        changed_pred[~mask] -= 1000
        self.assertEqual(float(masked_patch_mse(changed_pred, target, mask)), 16.5)
        loss.backward()
        self.assertTrue(torch.equal(pred.grad[~mask], torch.zeros_like(pred.grad[~mask])))
        self.assertTrue(bool((pred.grad[mask] != 0).all()))
        self.assertTrue(torch.equal(pred.grad[mask], pred.detach()[mask] / 2))
        self.assertIsNone(target.grad)

    def test_loss_rejects_invalid_and_nonfinite_inputs(self):
        pred = torch.ones(1, 2, 2, 2, 96)
        target = torch.zeros_like(pred)
        mask = make_mask(1, (2, 2, 2), 0.5, "spatial_tube")
        invalid_calls = [
            lambda: masked_patch_mse(pred, target[..., :1], mask),
            lambda: masked_patch_mse(pred, target, mask[:, :1]),
            lambda: masked_patch_mse(pred, target, torch.zeros_like(mask)),
            lambda: masked_patch_mse(pred[:, 0], target[:, 0], mask[:, 0]),
        ]
        for call in invalid_calls:
            with self.assertRaises(ValueError):
                call()
        with self.assertRaises(TypeError):
            masked_patch_mse(pred, target, mask.float())
        with self.assertRaises(TypeError):
            masked_patch_mse(pred.long(), target.long(), mask)
        for fill in (float("nan"), float("inf"), -float("inf")):
            for is_target in (False, True):
                invalid = target.clone() if is_target else pred.clone()
                invalid[~mask] = fill
                with self.assertRaises(ValueError):
                    masked_patch_mse(pred if is_target else invalid, invalid if is_target else target, mask)
        with self.assertRaises(ValueError):
            masked_patch_mse(pred * 1e30, target, mask)


def main():
    started = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(PrimitiveTests)
    test_names = [test.id().rsplit(".", 1)[-1] for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    source_files = (ROOT / "scripts" / "ssl_primitives.py", Path(__file__).resolve())
    record = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "scope": "Synthetic CPU tensor invariants only; no data, model, training, or paper-result claim",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "tests_run": result.testsRun,
        "test_names": test_names,
        "failures": [{"test": str(test), "traceback": error} for test, error in result.failures],
        "errors": [{"test": str(test), "traceback": error} for test, error in result.errors],
        "elapsed_seconds": time.perf_counter() - started,
        "source_sha256": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "patch_order": "dt, dh, dw, channel; RGB adjacent",
        "mask_true_means": "masked",
        "mask_modes": ["spatial_tube", "temporal_block", "random_3d"],
        "loss_choice": "Raw detached-target MSE; engineering diagnostic, not established as paper Eq. 13",
    }
    destination = Path(__file__).with_name("primitives_test_result.json")
    destination.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "tests_run": result.testsRun, "result": str(destination)}))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
