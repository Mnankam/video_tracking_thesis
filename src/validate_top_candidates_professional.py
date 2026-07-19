#!/usr/bin/env python3
"""
validate_top_candidates.py

Professional Stage 3 Wrapper
Author: Serge Kouomnankam

Features
--------
- Reuses validate_video_vs_imu()
- Batch validation
- Logging
- Resume support
- Ranking
- comparison.csv
- summary.json
- HTML summary
- Error isolation
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from datetime import datetime

from validate_video_vs_imu_v2 import ValidationConfig, validate_video_vs_imu

LOGGER=logging.getLogger("validate_top_candidates")


def load_candidates(csv_file):
    rows=[]
    with open(csv_file,newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        for i,row in enumerate(reader,1):
            rows.append({
                "rank":int(row.get("rank",i)),
                "imu_csv":row["imu_csv"],
                "label":row.get("label",Path(row["imu_csv"]).parent.name)
            })
    return rows


def html_summary(results,outfile):
    lines=[
        "<html><head><title>Candidate Summary</title></head><body>",
        "<h1>Validation Summary</h1>",
        "<table border='1'>",
        "<tr><th>Rank</th><th>Candidate</th><th>Pearson</th><th>RMSE</th><th>Lag[s]</th></tr>"
    ]
    for r in results:
        lines.append(
            f"<tr><td>{r['rank']}</td><td>{r['label']}</td>"
            f"<td>{r['pearson']}</td><td>{r['rmse']}</td><td>{r['lag_s']}</td></tr>"
        )
    lines.append("</table></body></html>")
    Path(outfile).write_text("\n".join(lines),encoding="utf-8")


def main():

    parser=argparse.ArgumentParser()

    parser.add_argument("--video-csv",required=True)
    parser.add_argument("--candidates-csv",required=True)
    parser.add_argument("--output-dir",required=True)

    parser.add_argument("--signal-column",default="inner_pipe_track_center_x")
    parser.add_argument("--imu-axis",default="liny")
    parser.add_argument("--resume",action="store_true")

    args=parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    results=[]
    failures=[]

    for cand in load_candidates(args.candidates_csv):

        run_dir=out/f"candidate_{cand['rank']:02d}_{cand['label']}"

        if args.resume and (run_dir/"validation_result.json").exists():
            LOGGER.info("Skipping %s",cand["label"])
            continue

        cfg=ValidationConfig(
            video_path=Path(args.video_csv),
            imu_path=Path(cand["imu_csv"]),
            output_dir=run_dir,
            video_value_column=args.signal_column,
            imu_value_column=args.imu_axis,
            run_name=f"Candidate {cand['rank']}",
        )

        try:
            res=validate_video_vs_imu(cfg)
            m=res.metrics
            results.append({
                "rank":cand["rank"],
                "label":cand["label"],
                "imu_csv":cand["imu_csv"],
                "pearson":m.get("pearson_correlation"),
                "rmse":m.get("rmse"),
                "mae":m.get("mae"),
                "r2":m.get("r_squared"),
                "lag_s":res.synchronization.get("lag_s"),
                "report":str(run_dir/"report"),
            })
        except Exception as exc:
            LOGGER.exception("Candidate failed")
            failures.append({
                "rank":cand["rank"],
                "candidate":cand["label"],
                "error":str(exc)
            })

    results.sort(key=lambda x:(-(x["pearson"] or -999),x["rmse"] or 1e9))

    if results:
        with open(out/"comparison.csv","w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)

    with open(out/"summary.json","w",encoding="utf-8") as f:
        json.dump({
            "created":datetime.now().isoformat(),
            "best":results[0] if results else None,
            "results":results,
            "failures":failures
        },f,indent=2,ensure_ascii=False)

    html_summary(results,out/"summary.html")

    print("="*70)
    print("FINAL RANKING")
    print("="*70)
    for r in results:
        print(f"{r['rank']:2d} {r['label']:25s} Pearson={r['pearson']:.4f} RMSE={r['rmse']:.4f}")

    if results:
        print("\\nBest candidate:",results[0]["label"])


if __name__=="__main__":
    main()