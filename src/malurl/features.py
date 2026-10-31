# -*- coding: utf-8 -*-
"""URL 特征提取（41 维）。

分四类：
    词法特征（10）：整体长度、主机/路径/查询长度，以及主域、顶级域、路径段的长度
    字符特征（15）：各类字符计数，覆盖整体、主机名、路径三个层面
    主机特征（8）：子域、IP、短链接、可疑顶级域、端口、品牌词位置
    结构统计（8）：路径深度、HTTPS、诱导词、熵值、可疑后缀

全部特征只依赖 URL 字符串，不发起任何网络请求。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

FEATURE_NAMES = [
    # ---- 词法（10）----
    "url_len", "host_len", "path_len", "query_len",
    "domain_len", "tld_len", "max_seg_len", "avg_seg_len",
    "n_segments", "n_params",
    # ---- 字符（15）----
    "n_digits", "n_letters", "n_dots", "n_hyphens",
    "n_underscore", "n_slash", "n_at", "n_question",
    "n_equal", "n_amp", "n_percent", "n_pct_encoded",
    "host_digits", "host_hyphens", "path_digits",
    # ---- 主机（8）----
    "n_subdomains", "has_ip", "has_at", "is_https",
    "is_shortener", "suspicious_tld", "has_port", "brand_in_subdomain",
    # ---- 结构（8）----
    "path_depth", "path_entropy", "path_hyphens", "suspicious_word_cnt",
    "digit_ratio", "host_entropy", "is_punycode", "suspicious_ext",
]

FEATURE_CN = {
    "url_len": "URL 总长度", "host_len": "主机名长度", "path_len": "路径长度",
    "query_len": "查询串长度", "domain_len": "主域长度", "tld_len": "顶级域长度",
    "max_seg_len": "最长路径段", "avg_seg_len": "平均路径段长度",
    "n_segments": "路径段数", "n_params": "参数个数",
    "n_digits": "数字个数", "n_letters": "字母个数", "n_dots": "点的个数",
    "n_hyphens": "连字符个数", "n_underscore": "下划线个数", "n_slash": "斜杠个数",
    "n_at": "@ 符号个数", "n_question": "问号个数", "n_equal": "等号个数",
    "n_amp": "& 符号个数", "n_percent": "百分号个数", "n_pct_encoded": "百分号编码数",
    "host_digits": "主机名数字数", "host_hyphens": "主机名连字符", "path_digits": "路径数字数",
    "n_subdomains": "子域层数", "has_ip": "是否 IP 直连", "has_at": "是否含 @",
    "is_https": "是否 HTTPS", "is_shortener": "是否短链接", "suspicious_tld": "可疑顶级域",
    "has_port": "是否带端口", "brand_in_subdomain": "品牌词仅在子域",
    "path_depth": "路径深度", "path_entropy": "路径信息熵", "path_hyphens": "路径连字符",
    "suspicious_word_cnt": "诱导词个数", "digit_ratio": "数字占比",
    "host_entropy": "主机名信息熵", "is_punycode": "是否 punycode",
    "suspicious_ext": "可疑文件后缀",
}

SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "work", "click",
    "link", "buzz", "loan", "rest", "fit", "icu", "cyou", "shop",
}

SHORTENERS = {
    "bit.ly", "t.cn", "goo.gl", "tinyurl.com", "ow.ly", "is.gd",
    "buff.ly", "rebrand.ly", "cutt.ly", "suo.im", "url.cn", "dwz.cn",
}

SUSPICIOUS_WORDS = [
    "login", "signin", "sign-in", "verify", "verification", "account",
    "update", "secure", "security", "confirm", "password", "passwd",
    "bank", "banking", "paypal", "alipay", "wallet", "webscr",
    "invoice", "unlock", "suspend", "recover", "billing", "auth",
]

SUSPICIOUS_EXT = {"exe", "zip", "rar", "apk", "scr", "bat", "cmd", "msi", "jar", "vbs", "ps1"}

# 常被仿冒的品牌词。用于判断"品牌词是否只出现在子域"——钓鱼站的典型手法：
# paypal.com.verify-login.tk 里 paypal 在子域，真正的域名却是 verify-login.tk
BRANDS = [
    "paypal", "apple", "microsoft", "google", "amazon", "netflix", "facebook",
    "instagram", "whatsapp", "alipay", "taobao", "wechat", "qq", "icbc",
    "ccb", "abc", "boc", "cmb", "bank", "alibaba", "jd", "baidu", "steam",
    "ebay", "dhl", "fedex", "dropbox", "adobe", "linkedin", "twitter",
]

_IP_RE = re.compile(r"^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$")
_PCT_RE = re.compile(r"%[0-9a-fA-F]{2}")

# 二级公共后缀：icbc.com.cn 的可注册域是 icbc.com.cn，不是 com.cn
# 不引入完整的 Public Suffix List（体积大），覆盖常见的中/日/英/澳等即可
TWO_LEVEL_SUFFIX = {
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn", "ac.cn",
    "co.uk", "org.uk", "gov.uk", "ac.uk", "me.uk",
    "co.jp", "ne.jp", "or.jp", "ac.jp",
    "com.tw", "com.hk", "com.au", "net.au", "com.sg", "com.my",
    "co.kr", "co.in", "com.br", "com.mx", "com.tr", "co.za", "co.nz",
}


def split_host(host: str):
    """把主机名拆成（可注册域, 二级域, 顶级域）。

    icbc.com.cn      -> ("icbc.com.cn", "icbc", "com.cn")
    ebank.cmbchina.com -> ("cmbchina.com", "cmbchina", "com")
    www.taobao.com   -> ("taobao.com", "taobao", "com")
    """
    if not host:
        return "", "", ""
    labels = host.split(".")
    if len(labels) >= 3 and ".".join(labels[-2:]) in TWO_LEVEL_SUFFIX:
        return ".".join(labels[-3:]), labels[-3], ".".join(labels[-2:])
    if len(labels) >= 2:
        return ".".join(labels[-2:]), labels[-2], labels[-1]
    return host, host, ""


def has_brand_outside_sld(host: str, sld: str) -> int:
    """品牌词是否只出现在子域（且是完整词，不是子串）。

    paypal.com.verify-login.tk -> paypal 在子域，真域名是 verify-login.tk -> 1
    ebank.cmbchina.com         -> "bank" 只是 "ebank" 的一部分 -> 0
    """
    if not host:
        return 0
    for b in BRANDS:
        start = host.find(b)
        while start != -1:
            left = host[start - 1] if start > 0 else ""
            right = host[start + len(b)] if start + len(b) < len(host) else ""
            if not (left.isalnum() or right.isalnum()):   # 词边界成立
                if b not in sld:
                    return 1
                break
            start = host.find(b, start + 1)
    return 0


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in Counter(s).values())


def normalize(url: str) -> str:
    """补全协议头，便于统一解析。"""
    url = str(url).strip()
    if not url:
        return ""
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    return url


def host_of(url: str) -> str:
    """取主机名，取不到时返回空串。分组切分要用它。"""
    try:
        return (urlparse(normalize(url)).hostname or "").lower()
    except ValueError:
        return ""


def extract_features(url: str) -> list:
    """输入一条 URL，返回 41 维特征（顺序与 FEATURE_NAMES 一致）。"""
    raw = str(url).strip()
    full = normalize(raw)
    if not full:
        return [0.0] * len(FEATURE_NAMES)

    try:
        parsed = urlparse(full)
    except ValueError:
        return [0.0] * len(FEATURE_NAMES)

    netloc = parsed.netloc or ""
    host = (parsed.hostname or "").lower()
    path = parsed.path or ""
    query = parsed.query or ""

    ip_match = _IP_RE.match(host)
    has_ip = 0
    if ip_match:
        has_ip = 1 if all(0 <= int(g) <= 255 for g in ip_match.groups()) else 0

    sld, domain, tld = split_host(host)

    segs = [s for s in path.split("/") if s]
    seg_lens = [len(s) for s in segs] or [0]

    ext = ""
    if segs and "." in segs[-1]:
        ext = segs[-1].rsplit(".", 1)[-1].lower()

    brand_in_sub = has_brand_outside_sld(host, sld)

    # 子域层数 = 主机标签数 - 可注册域标签数。
    # 不能直接用 host.count(".") - 1：icbc.com.cn 的可注册域本身有两级后缀，
    # 老算法会把"0 层子域"算成 1 层、把 www.icbc.com.cn 算成 2 层。
    if has_ip or not sld:
        n_subdomains = 0
    else:
        n_subdomains = max(
            len(host.rstrip(".").split(".")) - len(sld.split(".")), 0)

    # 是否带端口：用 urlparse 的 .port 判断。
    # 老写法 ":" in netloc 会把 user:pass@host 的冒号、
    # 以及 IPv6 的 [::1] 误判成"带端口"。
    try:
        has_port = 1.0 if parsed.port is not None else 0.0
    except ValueError:
        # 端口越界（如 :99999）或写成 :abc 时 .port 会抛 ValueError，
        # 但这类 netloc 确实带了端口分隔符。
        authority = netloc.rsplit("@", 1)[-1]
        has_port = 1.0 if ":" in authority.rsplit("]", 1)[-1] else 0.0

    lower_raw = raw.lower()
    n_digits = sum(ch.isdigit() for ch in raw)

    return [
        # ---- 词法（10）----
        float(len(raw)),
        float(len(host)),
        float(len(path)),
        float(len(query)),
        float(len(domain)),
        float(len(tld)),
        float(max(seg_lens)),
        float(sum(seg_lens) / len(seg_lens)),
        float(len(segs)),
        float(len([p for p in query.split("&") if p])),
        # ---- 字符（15）----
        float(n_digits),
        float(sum(ch.isascii() and ch.isalpha() for ch in raw)),
        float(raw.count(".")),
        float(raw.count("-")),
        float(raw.count("_")),
        float(raw.count("/")),
        float(raw.count("@")),
        float(raw.count("?")),
        float(raw.count("=")),
        float(raw.count("&")),
        float(raw.count("%")),
        float(len(_PCT_RE.findall(full))),
        float(sum(ch.isdigit() for ch in host)),
        float(host.count("-")),
        float(sum(ch.isdigit() for ch in path)),
        # ---- 主机（8）----
        float(n_subdomains),
        float(has_ip),
        1.0 if "@" in raw else 0.0,
        1.0 if full.lower().startswith("https://") else 0.0,
        1.0 if host in SHORTENERS else 0.0,
        1.0 if tld in SUSPICIOUS_TLDS else 0.0,
        has_port,
        float(brand_in_sub),
        # ---- 结构（8）----
        float(len(segs)),
        _entropy(path),
        float(path.count("-")),
        float(sum(1 for w in SUSPICIOUS_WORDS if w in lower_raw)),
        float(n_digits) / max(len(raw), 1),
        _entropy(host),
        1.0 if "xn--" in host else 0.0,
        1.0 if ext in SUSPICIOUS_EXT else 0.0,
    ]


def extract_many(urls) -> list:
    return [extract_features(u) for u in urls]
