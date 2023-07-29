# encoding: utf-8

import json
from collections import defaultdict

# prev_gt_ann_path = "/sdb/sjy/projects/actionformer_dev/data/multithumos/multithumos.json"
# generated_ann_pth = "/sdb/sjy/projects/actionformer_dev/data/multithumos/multithumos_processed.json"
prev_gt_ann_path = "/sdb/sjy/projects/actionformer_dev/data/charades/charades.json"
generated_ann_pth = "/sdb/sjy/projects/actionformer_dev/data/charades/charades_processed.json"

with open(prev_gt_ann_path, 'r', encoding='utf8') as f:
    json_db = json.load(f)

new_dict = defaultdict(str)
for video_id, v in json_db.items():
    annotations = []
    for item in v['actions']:
        annotations.append({
            "segment": [item[1], item[2]],
            # "label_id": int(item[0] - 1)        # for multithmos
            "label_id": int(item[0])
        })
    new_dict[video_id] = {
        "subset": v['subset'],
        "duration": v['duration'],
        "annotations": annotations
    }
with open(generated_ann_pth, 'w') as f:
    json.dump(new_dict, f)