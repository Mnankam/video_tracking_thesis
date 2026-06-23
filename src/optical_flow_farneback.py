import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import yaml


def load_config(path):
    with open(path,"r",encoding="utf-8") as f:
        return yaml.safe_load(f)


def resize_frame(frame,config):
    rw=config.get("resize_width")
    rh=config.get("resize_height")

    if rw and rh:
        return cv2.resize(frame,(int(rw),int(rh)))

    return frame


def preprocess(frame):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(5,5),0)
    gray=cv2.equalizeHist(gray)
    return gray


def crop_roi(arr,roi):

    if roi is None:
        return None

    x,y,w,h=map(int,roi)

    return arr[y:y+h,x:x+w]


def analyse_roi(flow,name,roi,frame):

    roi_flow=crop_roi(flow,roi)

    if roi_flow is None:
        return None

    dx=roi_flow[:,:,0]
    dy=roi_flow[:,:,1]

    mag=np.sqrt(dx**2+dy**2)

    return {
        "frame":frame,
        "roi_name":name,
        "mean_dx":float(np.mean(dx)),
        "mean_dy":float(np.mean(dy)),
        "mean_magnitude":float(np.mean(mag))
    }


def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--output-csv",default=None)

    args=parser.parse_args()

    config=load_config(args.config)

    video_path=config["video_path"]
    output_csv=args.output_csv or config["optical_flow_csv"]

    rois={
        "inner_pipe":config.get("inner_pipe_roi"),
        "bed_edge":config.get("bed_edge_roi"),
        "particle_bed":config.get("bed_roi")
    }

    cap=cv2.VideoCapture(video_path)

    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame=int(config.get("start_frame",0))
    end_frame=config.get("end_frame",total_frames)

    cap.set(cv2.CAP_PROP_POS_FRAMES,start_frame)

    ok,frame=cap.read()

    frame=resize_frame(frame,config)
    prev_gray=preprocess(frame)

    rows=[]
    processed_frames=0

    total_start=time.perf_counter()

    frame_idx=start_frame+1

    while frame_idx<end_frame:

        ok,frame=cap.read()

        if not ok:
            break

        frame=resize_frame(frame,config)

        gray=preprocess(frame)

        frame_start=time.perf_counter()

        flow=cv2.calcOpticalFlowFarneback(
            prev_gray,
            gray,
            None,
            0.5,
            3,
            25,
            3,
            7,
            1.5,
            0
        )

        frame_runtime=time.perf_counter()-frame_start

        for roi_name,roi in rois.items():

            result=analyse_roi(flow,roi_name,roi,frame_idx)

            if result:

                result["method"]="farneback_dense_cpu"
                result["compute_time_s"]=frame_runtime

                rows.append(result)

        prev_gray=gray.copy()

        processed_frames+=1
        frame_idx+=1

    total_runtime=time.perf_counter()-total_start

    cap.release()

    df=pd.DataFrame(rows)
    df.to_csv(output_csv,index=False)

    summary=pd.DataFrame([{
        "method":"farneback_dense_cpu",
        "processed_frames":processed_frames,
        "total_runtime_s":total_runtime,
        "avg_frame_time_s":total_runtime/processed_frames,
        "effective_fps":processed_frames/total_runtime
    }])

    benchmark=output_csv.replace(".csv","_benchmark.csv")
    summary.to_csv(benchmark,index=False)

    print("Farneback abgeschlossen")
    print(summary)