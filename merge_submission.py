import json
import os

result_det = "submission.json"
result_rev = result_det
submission_file = "submission_final.json"

with open(result_det, 'r') as fobj:
    data_det = json.load(fobj)

with open(result_rev, 'r') as fobj:
    data_rev = json.load(fobj)

data_submission = {"version": "1.0", "challenge": "ego4d_moment_queries"}

data_submission['detect_results'] = data_det['results']
data_submission['retrieve_results'] = data_rev['results']

with open(submission_file, "w") as fp:
    json.dump(data_submission, fp)

    