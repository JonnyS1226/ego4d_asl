# python imports
import argparse
import os
import glob
import time
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import valid_one_epoch, ANETdetection, fix_random_seed, infer_one_epoch_ensemble
import logging

################################################################################
def main(args):
    """0. load config"""
    # sanity check
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    assert len(cfg['val_split']) > 0, "Test set must be specified!"

    ckpt_file1 = args.ckpt1
    ckpt_file2 = args.ckpt2
    ckpt_file3 = args.ckpt3
    ckpt_file4 = args.ckpt4
    ckpt_file5 = args.ckpt5
        
    log_dir = "logs/ensemble/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(log_dir, "log.txt"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)
    
    if args.topk > 0:
        cfg['model']['test_cfg']['max_seg_num'] = args.topk
    pprint(cfg)

    """1. fix all randomness"""
    # fix the random seeds (this will fix everything)
    _ = fix_random_seed(0, include_cuda=True)

    """2. create dataset / dataloader"""
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    # set bs = 1, and disable shuffle
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )

    """3. create model and evaluator"""
    # model
    model1 = make_meta_arch(cfg['model_name'], **cfg['model'])
    model2 = make_meta_arch(cfg['model_name'], **cfg['model'])
    model3 = make_meta_arch(cfg['model_name'], **cfg['model'])
    model4 = make_meta_arch(cfg['model_name'], **cfg['model'])
    model5 = make_meta_arch(cfg['model_name'], **cfg['model'])
    # model6 = make_meta_arch(cfg['model_name'], **cfg['model'])
    # model7 = make_meta_arch(cfg['model_name'], **cfg['model'])
    # model8 = make_meta_arch(cfg['model_name'], **cfg['model'])
    # # not ideal for multi GPU training, ok for now
    model1 = nn.DataParallel(model1, device_ids=cfg['devices'])
    model2 = nn.DataParallel(model2, device_ids=cfg['devices'])
    model3 = nn.DataParallel(model3, device_ids=cfg['devices'])
    model4 = nn.DataParallel(model4, device_ids=cfg['devices'])
    model5 = nn.DataParallel(model5, device_ids=cfg['devices'])
    # model6 = nn.DataParallel(model6, device_ids=cfg['devices'])
    # model7 = nn.DataParallel(model7, device_ids=cfg['devices'])
    # model8 = nn.DataParallel(model8, device_ids=cfg['devices'])

    """4. load all ckpts"""
    print("=> loading checkpoint '{}'".format(ckpt_file1))
    # load ckpt, reset epoch / best rmse
    checkpoint1 = torch.load(
        ckpt_file1, map_location="cpu"
    )
    # load ema model instead
    print("Loading from EMA model1 ...")
    model1.load_state_dict(checkpoint1['state_dict_ema'])
    del checkpoint1

    checkpoint2 = torch.load(
        ckpt_file2, map_location="cpu"
    )
    # load ema model instead
    print("Loading from EMA model2 ...")
    model2.load_state_dict(checkpoint2['state_dict_ema'])
    del checkpoint2
    
    checkpoint3 = torch.load(
        ckpt_file3, map_location="cpu"
    )
    # load ema model instead
    print("Loading from EMA model3 ...")
    model3.load_state_dict(checkpoint3['state_dict_ema'])
    del checkpoint3

    checkpoint4 = torch.load(
        ckpt_file4, map_location="cpu"
    )
    # load ema model instead
    print("Loading from EMA model4 ...")
    model4.load_state_dict(checkpoint4['state_dict_ema'])
    del checkpoint4

    checkpoint5 = torch.load(
        ckpt_file5, map_location="cpu"
    )
    # load ema model instead
    print("Loading from EMA model5 ...")
    model5.load_state_dict(checkpoint5['state_dict_ema'])
    del checkpoint5
    
    # set up evaluator
    det_eval, output_file = None, None
    val_db_vars = val_dataset.get_attributes()
    det_eval = ANETdetection(
        val_dataset.json_file,
        val_dataset.split[0],
        tiou_thresholds = val_db_vars['tiou_thresholds']
    )

    """5. Test the model"""
    print("\nStart testing model {:s} ...".format(cfg['model_name']))
    start = time.time()
    
    _ = infer_one_epoch_ensemble(
        val_loader,
        model1,
        model2,
        model3,
        model4,
        model5,
        -1,
        evaluator=det_eval,
        output_file=output_file,
        ext_score_file=cfg['test_cfg']['ext_score_file'],
        tb_writer=None,
        print_freq=args.print_freq,
        logger=logger
    )

    end = time.time()
    print("All done! Total time: {:0.2f} sec".format(end - start))
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization')
    parser.add_argument('config', type=str, metavar='DIR',
                        help='path to a config file')
    parser.add_argument('--ckpt1', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('--ckpt2', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('--ckpt3', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('--ckpt4', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('--ckpt5', type=str, metavar='DIR',
                        help='path to a checkpoint')
    parser.add_argument('-t', '--topk', default=-1, type=int,
                        help='max number of output actions (default: -1)')
    parser.add_argument('--saveonly', action='store_true',
                        help='Only save the ouputs without evaluation (e.g., for test set)')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    args = parser.parse_args()
    main(args)