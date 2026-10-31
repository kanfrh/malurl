# -*- coding: utf-8 -*-
"""特征提取的边界测试：这些地方最容易在真实流量上崩。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from malurl.features import (FEATURE_CN, FEATURE_NAMES, extract_features,
                             host_of, normalize)


def test_feature_dimension():
    feats = extract_features("https://www.example.com/a?b=1")
    assert len(feats) == 41 == len(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in feats)


def test_all_features_have_chinese_name():
    assert set(FEATURE_CN) == set(FEATURE_NAMES)


def test_empty_url_returns_zeros():
    assert extract_features("") == [0.0] * 41
    assert extract_features("   ") == [0.0] * 41


def test_missing_scheme_is_added():
    assert normalize("www.example.com") == "http://www.example.com"
    assert normalize("https://a.com") == "https://a.com"


def test_https_flag():
    assert extract_features("https://a.com")[FEATURE_NAMES.index("is_https")] == 1.0
    assert extract_features("http://a.com")[FEATURE_NAMES.index("is_https")] == 0.0


def test_ip_host_detected():
    idx = FEATURE_NAMES.index("has_ip")
    assert extract_features("http://192.168.1.1/admin")[idx] == 1.0
    assert extract_features("http://999.1.1.1/admin")[idx] == 0.0


def test_at_sign_trick():
    idx = FEATURE_NAMES.index("has_at")
    assert extract_features("http://paypal.com@evil.tk/login")[idx] == 1.0


def test_suspicious_tld_and_words():
    tld = FEATURE_NAMES.index("suspicious_tld")
    words = FEATURE_NAMES.index("suspicious_word_cnt")
    feats = extract_features("http://secure-paypal.tk/login/verify")
    assert feats[tld] == 1.0
    assert feats[words] >= 2


def test_unicode_and_long_url_do_not_crash():
    assert len(extract_features("http://例子.中国/登录?q=测试")) == 41
    assert len(extract_features("http://a.com/" + "x" * 5000)) == 41


def test_garbage_input_does_not_crash():
    for bad in ["://", "http://", "http://[::1]:8080/x", "@@@", "\x00\x01"]:
        assert len(extract_features(bad)) == 41


def test_host_of():
    assert host_of("https://WWW.Example.COM/a") == "www.example.com"
    assert host_of("http://1.2.3.4/x") == "1.2.3.4"
    assert host_of("") == ""
