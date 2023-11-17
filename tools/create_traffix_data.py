# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from os import path as osp

from tools.data_converter.create_traffix_gt_database import (create_groundtruth_database)

def traffix_data_prep(root_path,
                 info_prefix,
                 out_dir,
                 workers):
    from tools.data_converter import traffix_converter as traffix
    # TODO: for inference just use testing
    splits = ['training', 'validation', 'testing']
    load_dir = osp.join(root_path)
    save_dir = osp.join(out_dir)
    os.makedirs(save_dir, exist_ok=True, mode=0o777)
    
    converter = traffix.traffix2Nuscenes(splits, load_dir, save_dir)
    converter.convert()
    # TODO: for inference do not run this
    create_groundtruth_database("TraffixNuscDataset", save_dir, info_prefix, f'{save_dir}/{info_prefix}_infos_train.pkl')


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/traffix',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/traffix_processed',
    required=False,
    help='name of info pkl')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    traffix_data_prep(
        root_path=args.root_path,
        info_prefix='traffix_nusc',
        out_dir=args.out_dir,
        workers=args.workers)
