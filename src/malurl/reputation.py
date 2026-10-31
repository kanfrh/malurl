# -*- coding: utf-8 -*-
"""域名信誉层（白名单）。

为什么需要它：
    机器学习模型擅长判断"没见过的"域名，但对"已知可信"的域名没必要冒险——
    工商银行登录页被误判一次，对用户体验的伤害远大于多抓一个钓鱼站。

分工：
    模型  负责未知域名，做结构判断；
    白名单 负责已知可信域名，直接放行；
    两者互不干扰，白名单命中时模型根本不参与计算。

匹配规则用"后缀 + 点"：
    www.icbc.com.cn     命中 icbc.com.cn     ✓
    icbc.com.cn         命中 icbc.com.cn     ✓
    icbc.com.cn.evil.tk 不命中               ✓（钓鱼常见手法）
    fakeicbc.com.cn     不命中               ✓
"""
from __future__ import annotations

from .features import host_of

# 只收录"登录页高频出现、误判代价极高"的域名
TRUSTED_DOMAINS = [
    # ---- 国内银行 / 支付 ----
    "icbc.com.cn", "ccb.com", "abchina.com", "bankofchina.com", "boc.cn",
    "cmbchina.com", "bankcomm.com", "spdb.com.cn", "cib.com.cn", "pingan.com",
    "cgbchina.com.cn", "hxb.com.cn", "psbc.com", "ecitic.com",
    "bankofbeijing.com.cn", "njcb.com.cn", "srcb.com",
    "alipay.com", "95516.com", "tenpay.com", "unionpay.com",
    # ---- 邮箱 / 门户 / 电商 ----
    "163.com", "126.com", "qq.com", "sina.com.cn", "baidu.com", "sohu.com",
    "aliyun.com", "jd.com", "taobao.com", "weibo.com", "zhihu.com",
    # ---- 政务 / 公共服务 ----
    "gov.cn", "12306.cn", "chinatax.gov.cn", "mps.gov.cn", "mof.gov.cn",
    "miit.gov.cn", "customs.gov.cn", "court.gov.cn",
    # ---- 高校 ----
    "tsinghua.edu.cn", "pku.edu.cn", "nju.edu.cn", "zju.edu.cn",
    "ustc.edu.cn", "fudan.edu.cn", "sjtu.edu.cn",
    # ---- 国外主流 ----
    "google.com", "microsoft.com", "apple.com", "amazon.com", "paypal.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "citibank.com",
    "hsbc.com", "dropbox.com", "github.com", "linkedin.com", "adobe.com",
]


def is_trusted(host: str) -> bool:
    """主机名是否属于白名单域名（含其子域）。"""
    host = (host or "").lower().strip(".")
    if not host:
        return False
    for d in TRUSTED_DOMAINS:
        if host == d or host.endswith("." + d):
            return True
    return False


def check(url: str) -> dict:
    """返回 {"trusted": bool, "host": str, "reason": str}。"""
    host = host_of(url)
    if is_trusted(host):
        return {"trusted": True, "host": host, "reason": "域名在可信白名单中"}
    return {"trusted": False, "host": host, "reason": ""}
