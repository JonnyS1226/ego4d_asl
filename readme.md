# Ego4D-ASL
[Techical report](https://arxiv.org/abs/2306.09172) | 1-st in MQ challenge and 2-nd in NLQ challenge in Ego4D workshop at CVPR 2023.

This report presents ReLER submission to two tracks in the Ego4D Episodic Memory Benchmark in CVPR 2023, including Natural Language Queries and Moment Queries. This solution inherits from our proposed Action Sensitivity Learning framework (ASL) to better capture discrepant information of frames. Further, we incorporate a series of stronger video features and fusion strategies. Our method achieves an average mAP of 29.34, ranking 1st in Moment Queries Challenge, and garners 19.79 mean R1, ranking 2nd in Natural Language Queries Challenge. Our code will be released.



## Changelog
* May have some bugs or problems, we are in progress to improve the released codes.
- [x] release the code for MQ
- [ ] release the code for NLQ
- [ ] tidy the code

## Installation
* GCC, PyTorch==1.12.0, CUDA==cu113 dependencies
* pip dependencies
```
conda env create -n py38 python==3.8
conda activate py38
pip install  tensorboard numpy pyyaml pandas h5py joblib
```
* NMS compilation
```
cd ./libs/utils
python setup.py install --user
cd ../..
```

## Data Preparation
* Ego4D MQ Annotation, Video Data / Features Preparation
    *   Please refer to [Ego4D website](https://ego4d-data.org/) to download features.
    *   In our submission, we finally use [InternVideo](https://arxiv.org/abs/2211.09529), [EgoVLP](https://github.com/showlab/EgoVLP), Slowfast and Omnivore features, where only combination of InternVideo and EgoVLP can achieve good results.

* Ego4D Video Features Preparation
    * By using `python convert_annotation.py` to convert official annotation to the processed one. And put it into `data/ego4d`. 
    * Create config file such as `baseline.yaml` corrsponding to training. And put it into `configs/`
    * In `baseline.yaml`, you can specify annotation json_file, video features, training split and validation split, e.t.c.

## Train on MQ (train-set)
* Change the train_split as `['train']` and val_split as `['val']`.
* ```bash train_val.sh baseline 0 ``` where `baseline` is the corresponding config yaml and `0` is the GPU ordinal.

## Validate on MQ (val-set)
* When running ```bash train_val.sh baseline 0 ```, in epoch > max_epoch //3, it will automatically validate performance on val-set (e.g., average mAP, Recall@1x)
* Can also run `bash val.sh checkpoint.ckpt baseline` to validate performance manually.
* It is expected to get average mAP between 27-28(%).

## Train on MQ (Train + val)
* Change the train_split as `['train', 'val]` and val_split as `['test']`.
* ```bash train_combine.sh baseline 0 ``` where `baseline` is the corresponding config yaml and `0` is the GPU ordinal.
* In this way, it will not validate performance during training, will save checkpoint of the last 5 epochs instead.

## Submission (to Leaderboard test-set server)
* `python infer.py --config configs/baseline.yaml --ckpt your_checkpoint` to finally generate `submission.json` of detection results.
* Then `python merge_submission.py` to generate `submission_final.json` which is results of both detection and retrieval.
* Upload `submission_final.json` to the [Ego4D MQ test-server](https://eval.ai/web/challenges/challenge-page/1626/leaderboard)


## Acknowledgement
Our model are based on [Actionformer](https://github.com/happyharrycn/actionformer_release/tree/main). Thanks for their contributions.


## Cite
```
@article{shao2023action,
  title={Action Sensitivity Learning for the Ego4D Episodic Memory Challenge 2023},
  author={Shao, Jiayi and Wang, Xiaohan and Quan, Ruijie and Yang, Yi},
  journal={arXiv preprint arXiv:2306.09172},
  year={2023}
}

@article{shao2023action,
  title={Action Sensitivity Learning for Temporal Action Localization},
  author={Shao, Jiayi and Wang, Xiaohan and Quan, Ruijie and Zheng, Junjun and Yang, Jiang and Yang, Yi},
  journal={arXiv preprint arXiv:2305.15701},
  year={2023}
}

```






