# encoding: utf-8
import json
from tqdm import tqdm
if __name__ == "__main__":
    ens1 = "submission.json"
    ens2 = "submission1.json"
    ens3 = "submission2.json"

    with open(ens1, 'r') as fobj1:
        det1 = json.load(fobj1)
    with open(ens2, 'r') as fobj2:
        det2 = json.load(fobj2)
    with open(ens3, 'r') as fobj3:
        det3 = json.load(fobj3)
    for k in tqdm(det2['results'].keys()):
        det1['results'][k] = det1['results'][k] + det2['results'][k] + det3['results'][k]
        
    data_submission = {"version": "1.0", "challenge": "ego4d_moment_queries"}

    data_submission['detect_results'] = det1['results']
    data_submission['retrieve_results'] = det1['results']

    submission_file = "submission_final.json"
    with open(submission_file, "w") as fp:
        json.dump(data_submission, fp)