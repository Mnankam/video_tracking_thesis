!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from validate_video_vs_imu_v2 import ValidationConfig, validate_video_vs_imu

def load_candidates(path:Path):
    with path.open("r",encoding="utf-8") as f:
        r=csv.DictReader(f)
        out=[]
        for i,row in enumerate(r,1):
            out.append({"rank":int(row.get("rank",i)),"imu_csv":row["imu_csv"]})
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--video-csv",required=True)
    p.add_argument("--candidates-csv",required=True)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--signal-column",default="inner_pipe_track_center_x")
    p.add_argument("--imu-axis",default="liny")
    a=p.parse_args()
    outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    comp=[]
    for c in load_candidates(Path(a.candidates_csv)):
        imu=Path(c["imu_csv"])
        run=outdir/f"candidate_{c['rank']:02d}_{imu.parent.name}"
        cfg=ValidationConfig(video_path=Path(a.video_csv),imu_path=imu,output_dir=run,
                             video_value_column=a.signal_column,
                             imu_value_column=a.imu_axis,
                             run_name=f"Candidate {c['rank']}")
        res=validate_video_vs_imu(cfg)
        m=res.metrics
        comp.append({"rank":c["rank"],"imu_csv":str(imu),
                     "pearson":m.get("pearson_correlation"),
                     "rmse":m.get("rmse"),"mae":m.get("mae"),
                     "lag_s":res.synchronization.get("lag_s"),
                     "report_dir":str(run)})
    comp=sorted(comp,key=lambda x:(-(x["pearson"] or -999),x["rmse"] or 1e9))
    with (outdir/"comparison.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(comp[0].keys()));w.writeheader();w.writerows(comp)
    with (outdir/"summary.json").open("w",encoding="utf-8") as f:
        json.dump(comp,f,indent=2,ensure_ascii=False)
    print("Best:",comp[0]["imu_csv"])
if __name__=="__main__":
    main()