"""
evaluate_events.py
Compute precision and recall for detected events against ground truth labels.
Expected ground truth CSV columns: video,event,time_s
Detected event CSVs are produced by drowsiness_demo (in outputs/events/)
"""

import pandas as pd
import os
import glob

DETECTED_DIR = "../outputs/events"
GROUND_TRUTH = "../data/labels_ground_truth.csv"   # you should create this

TIME_TOLERANCE_S = 1.0   # tolerance in seconds when matching events

def load_detected_all():
    files = glob.glob(os.path.join(DETECTED_DIR, "*_events.csv"))
    dfs = []
    for f in files:
        d = pd.read_csv(f)
        # ensure columns: time_s, event
        if 'time_s' in d.columns:
            d['video'] = os.path.basename(f).replace("_events.csv","")
            dfs.append(d[['video','event','time_s']])
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame(columns=['video','event','time_s'])

def evaluate(gt_df, det_df):
    # For each GT event, find if there's a detected event of same type within tolerance
    matched_gt_idx = set()
    matched_det_idx = set()

    for i, gt in gt_df.iterrows():
        vid = gt['video']
        ev = gt['event']
        t = gt['time_s']
        # search detections in same video and type
        cand = det_df[(det_df['video']==vid) & (det_df['event']==ev)]
        if cand.empty:
            continue
        # compute time diffs
        cand['td'] = (cand['time_s'] - t).abs()
        close = cand[cand['td'] <= TIME_TOLERANCE_S]
        if not close.empty:
            # match the earliest close detection
            j = close['td'].idxmin()
            matched_gt_idx.add(i)
            matched_det_idx.add(j)

    TP = len(matched_gt_idx)
    FN = len(gt_df) - TP
    FP = len(det_df) - len(matched_det_idx)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return {"TP":TP,"FP":FP,"FN":FN,"precision":precision,"recall":recall}

def main():
    gt = pd.read_csv(GROUND_TRUTH)
    detections = load_detected_all()
    res = evaluate(gt, detections)
    print("Evaluation results:", res)
    pd.DataFrame([res]).to_csv("../outputs/eval_summary.csv", index=False)

if __name__ == "__main__":
    main()
