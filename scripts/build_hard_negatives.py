# -*- coding: utf-8 -*-
"""构造"硬负样本"：真实存在、天然带 login 路径的正常站点。

背景：训练集里正常样本 4 万条，路径含 login 类词的只有 357 条（0.89%），
而恶意样本里这一比例是 9.68%——模型因此把"登录词"学成了强信号，
导致工商银行、招商银行这类真实登录页被误判。

做法：挑选各行业真实存在的登录入口域名，按常见路径拼出 URL，标为正常样本。
这些 URL 是"真实域名 + 常见路径"的半合成数据，作用是让模型看到
"干净的域名 + login 路径"这个组合，从而不再单凭登录词下判断。

用法：
    python scripts/build_hard_negatives.py --out data/hard_negatives.csv
    python scripts/build_hard_negatives.py --merge data/urls.csv --out data/urls_v2.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# 国内银行 / 支付
CN_FINANCE = [
    "icbc.com.cn", "ccb.com", "abchina.com", "bankofchina.com", "boc.cn",
    "cmbchina.com", "bankcomm.com", "spdb.com.cn", "cib.com.cn", "pingan.com",
    "cgbchina.com.cn", "hxb.com.cn", "psbc.com", "ecitic.com",
    "bankofbeijing.com.cn", "njcb.com.cn", "srcb.com", "alipay.com",
    "95516.com", "tenpay.com", "unionpay.com", "hexun.com",
]

# 邮箱 / 门户 / 社交（都有真实登录页）
CN_PORTAL = [
    "163.com", "126.com", "qq.com", "sina.com.cn", "baidu.com", "sohu.com",
    "aliyun.com", "jd.com", "taobao.com", "weibo.com", "zhihu.com", "douban.com",
]

# 政务 / 公共服务
CN_GOV = [
    "gov.cn", "12306.cn", "chinatax.gov.cn", "mps.gov.cn", "mof.gov.cn",
    "miit.gov.cn", "customs.gov.cn", "stats.gov.cn", "court.gov.cn",
]

# 高校
CN_EDU = [
    "tsinghua.edu.cn", "pku.edu.cn", "nju.edu.cn", "zju.edu.cn",
    "ustc.edu.cn", "fudan.edu.cn", "sjtu.edu.cn", "ruc.edu.cn",
]

# 国外主流站点（登录页最典型的一批）
GLOBAL = [
    "google.com", "microsoft.com", "apple.com", "amazon.com", "paypal.com",
    "chase.com", "bankofamerica.com", "wellsfargo.com", "citibank.com",
    "hsbc.com", "barclays.co.uk", "dropbox.com", "github.com", "linkedin.com",
    "facebook.com", "netflix.com", "steampowered.com", "ebay.com", "adobe.com",
]

ROOT_PATHS = ["/login", "/signin", "/user/login", "/account/login", "/sso/login"]
BANK_PATHS = ["/login", "/signin", "/user/login", "/account/login", "/sso/login",
              "/ebank/login", "/personal/login", "/corporate/login"]
SUB_HOSTS = ["login", "passport", "auth", "sso", "ebank", "my"]


def build() -> pd.DataFrame:
    urls = set()

    def add(u):
        urls.add(u)

    for group, paths in ((CN_FINANCE, BANK_PATHS), (CN_PORTAL, ROOT_PATHS),
                         (CN_GOV, ROOT_PATHS), (CN_EDU, ROOT_PATHS),
                         (GLOBAL, ROOT_PATHS)):
        for d in group:
            for p in paths:
                add("https://www.%s%s" % (d, p))
                add("https://%s%s" % (d, p))
            for sub in SUB_HOSTS[:3]:
                add("https://%s.%s/" % (sub, d))

    # 显式补上本次要解释的案例
    for u in [
        "https://www.icbc.com.cn/login",
        "https://www.icbc.com.cn/ICBC/",
        "https://ebank.cmbchina.com/login",
        "https://mail.163.com/",
        "https://passport.baidu.com/v2/?login",
        "https://login.microsoftonline.com/",
        "https://accounts.google.com/signin",
        "https://github.com/login",
    ]:
        add(u)

    df = pd.DataFrame(sorted(urls), columns=["url"])
    df["label"] = 0
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/hard_negatives.csv")
    ap.add_argument("--merge", default="", help="与已有数据集合并，输出新文件")
    args = ap.parse_args()

    hard = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 硬负样本单独存一份，便于复查
    hard_path = Path("data/hard_negatives.csv")
    hard_path.parent.mkdir(parents=True, exist_ok=True)
    hard.to_csv(hard_path, index=False, encoding="utf-8-sig")
    domains = hard["url"].str.extract(r"https?://([^/]+)")[0].nunique()
    print("硬负样本 %d 条，覆盖域名 %d 个（已存 %s）" % (len(hard), domains, hard_path))

    if args.merge:
        base = pd.read_csv(args.merge, dtype=str)
        base["url"] = base["url"].astype(str).str.strip()
        merged = pd.concat([base[["url", "label"]], hard[["url", "label"]]], ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset=["url"])
        merged["label"] = merged["label"].astype(int)
        merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
        merged.to_csv(out, index=False, encoding="utf-8-sig")
        print("合并后 %d 条（原 %d + 硬负样本 %d，去重去掉 %d 条）"
              % (len(merged), len(base), len(hard), before - len(merged)))
        print("已写入 %s" % out)
    else:
        hard.to_csv(out, index=False, encoding="utf-8-sig")
        print("已写入 %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
