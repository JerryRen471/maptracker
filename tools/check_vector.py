import json, numpy as np
p = "/data/maptr_workspace/work_dirs/debug_overfit6_stage1_front_roi/submission_vector.json"
with open(p) as f:
    r = json.load(f)["results"]

scores = []
counts = []
for token, pred in r.items():
    s = pred.get("scores", [])
    scores.extend(s)
    counts.append(len(s))

print("tokens:", len(r))
print("pred count per frame min/med/max:", min(counts), np.median(counts), max(counts))
if scores:
    scores = np.asarray(scores, dtype=float)
    print("score min/p50/p90/p99/max:", np.min(scores), np.percentile(scores, 50),
    np.percentile(scores, 90), np.percentile(scores, 99), np.max(scores))
    print("num >=0.05:", int((scores >= 0.05).sum()))
    print("num >=0.01:", int((scores >= 0.01).sum()))
else:
    print("NO SCORES")
