# -*- coding: utf-8 -*-
"""命令行入口：

    malurl train   --data data/urls.csv
    malurl compare --data data/urls.csv      # 随机切分 vs 分组切分
    malurl predict "http://a.tk/login"
    malurl drift   --data data/urls.csv
    malurl replay  --limit 200
    malurl serve
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import get_settings
from .data import load_dataset
from .features import FEATURE_NAMES, extract_features
from .model import compare_splits, load, predict_proba_safe, train
from . import store

settings = get_settings()


def _fmt_table(df) -> str:
    return df.to_string(index=False, float_format=lambda v: "%.4f" % v)


def cmd_train(args) -> int:
    df = load_dataset(args.data)
    print("样本：%d（正常 %d / 恶意 %d）"
          % (len(df), (df.label == 0).sum(), (df.label == 1).sum()))
    info = train(df, settings.resolve(Path(args.out)), data_file=Path(args.data).name)
    print("\n选定模型：%s" % info["model_name"])
    print("理由：%s" % info["select_reason"])
    print("测试集指标：")
    for k, v in info["metrics"].items():
        print("  %-10s %.4f" % (k, v))
    print("\n模型已保存：%s" % (settings.resolve(Path(args.out)) / "model.joblib"))
    return 0


def cmd_compare(args) -> int:
    df = load_dataset(args.data)
    table = compare_splits(df)
    print(table.to_string(index=False, float_format=lambda v: "%.4f" % v))
    out = settings.resolve(Path(args.out)) / "split_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print("\n已保存：%s" % out)
    rnd, grp = table.iloc[0], table.iloc[1]
    gap = (rnd["accuracy"] - grp["accuracy"]) * 100
    print("\n随机切分的域名重叠率 %.1f%%，分组切分恒为 0%%。" % (rnd["domain_overlap"] * 100))
    print("两种切分的准确率相差 %+.2f 个百分点。" % gap)
    if gap > 0.5:
        print("→ 随机切分明显高估，分组切分的结果才可信。")
    elif gap < -0.5:
        print("→ 分组切分反而更高，说明模型没有依赖域名身份，这是好消息。")
    else:
        print("→ 两者基本持平：特征本身不依赖域名身份，评估结论稳定。")
    return 0


def cmd_predict(args) -> int:
    bundle = load(settings.resolve(settings.model_path))
    for url in args.urls:
        x = np.asarray(extract_features(url), dtype=float).reshape(1, -1)
        score = float(predict_proba_safe(bundle["model"], x)[0, 1])
        label = "恶意" if score >= settings.threshold else "正常"
        print("%-60s %7.2f%%  %s" % (url[:60], score * 100, label))
    return 0


def cmd_drift(args) -> int:
    import joblib
    from .drift import drift_report

    ref_path = settings.resolve(Path(args.out)) / "psi_reference.joblib"
    if not ref_path.exists():
        print("找不到训练集分布文件：%s\n请先运行 malurl train" % ref_path)
        return 2
    reference = joblib.load(ref_path)
    df = load_dataset(args.data)
    X = np.asarray([extract_features(u) for u in df["url"]], dtype=float)
    report = drift_report(reference, X)
    print("最大 PSI：%.4f（%s）→ %s" % (report["max_psi"], report["level"], report["advice"]))
    for item in report["top"]:
        print("  %-24s %-10s %.4f" % (item["feature"], item["name_cn"], item["psi"]))
    return 0


def cmd_replay(args) -> int:
    rows = store.read_recent(settings.resolve(settings.log_path), args.limit)
    if not rows:
        print("没有判定日志。")
        return 0
    n_mal = sum(1 for r in rows if r.get("label") == "malicious")
    print("最近 %d 条判定：恶意 %d 条（%.1f%%）"
          % (len(rows), n_mal, 100.0 * n_mal / len(rows)))
    print("最早：%s   最新：%s" % (rows[0].get("ts"), rows[-1].get("ts")))
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    uvicorn.run("malurl.service.api:app", host=settings.host,
                port=args.port or settings.port, reload=False)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="malurl", description="恶意 URL 检测服务")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="训练模型")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="artifacts")
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("compare", help="对比随机切分与分组切分")
    c.add_argument("--data", required=True)
    c.add_argument("--out", default="artifacts")
    c.set_defaults(func=cmd_compare)

    pr = sub.add_parser("predict", help="命令行判定")
    pr.add_argument("urls", nargs="+")
    pr.set_defaults(func=cmd_predict)

    d = sub.add_parser("drift", help="计算特征漂移")
    d.add_argument("--data", required=True)
    d.add_argument("--out", default="artifacts")
    d.set_defaults(func=cmd_drift)

    r = sub.add_parser("replay", help="统计判定日志")
    r.add_argument("--limit", type=int, default=1000)
    r.set_defaults(func=cmd_replay)

    s = sub.add_parser("serve", help="启动 HTTP 服务")
    s.add_argument("--port", type=int, default=0)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
