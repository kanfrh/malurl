# -*- coding: utf-8 -*-
"""PSI 漂移的边界测试：训练分布之外的样本不能被静默丢弃。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from malurl.drift import compute_psi, drift_report  # noqa: E402
from malurl.model import build_psi_reference  # noqa: E402


def test_psi_is_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    values = compute_psi(build_psi_reference(X), X)
    assert len(values) == 3
    assert max(values) < 1e-6


def test_psi_detects_values_outside_training_range():
    """全部样本跑到训练范围外，PSI 必须报显著漂移，而不是接近 0。"""
    X = np.linspace(0.0, 1.0, 500).reshape(-1, 1)
    reference = build_psi_reference(X)
    far = np.full((500, 1), 1000.0)

    values = compute_psi(reference, far)
    assert values[0] > 0.25
    assert drift_report(reference, far)["level"] == "显著漂移"
