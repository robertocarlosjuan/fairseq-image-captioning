import argparse
import base64
import csv
import os
import sys
import torch
import tqdm
import numpy as np


csv.field_size_limit(sys.maxsize)

FIELDNAMES_IN = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
FIELDNAMES_OUT = FIELDNAMES_IN[:-2]

def check_if_zeroone(train_custom):
    if os.path.isfile(train_custom):
        train_list = [train_custom]
    else:
        train_list = [train_custom + '.0', train_custom + '.1']
        for x in train_list:
            if not os.path.isfile(x):
                return None
    return train_list

def features_files(features_dir, args):
    train_custom = os.path.join(features_dir,'karpathy_train_resnet101_faster_rcnn_genome.tsv')
    train_list = check_if_zeroone(train_custom)
    valid_file = os.path.join(features_dir, f'karpathy_val_resnet101_faster_rcnn_genome.tsv')
    valid_list = check_if_zeroone(valid_file)
    if not valid_list:
        valid_file = os.path.join(features_dir, f'karpathy_valid_resnet101_faster_rcnn_genome.tsv')
        valid_list = check_if_zeroone(valid_file)
    
    test_list = check_if_zeroone(os.path.join(features_dir, f'karpathy_test_resnet101_faster_rcnn_genome.tsv'))
    return {
        'train': train_list,
        'valid': valid_list,
        'test': test_list,
        'custom': [args.custom_feature_dir]
    }


def main(args):
    in_features_files = features_files(args.features_dir, args)[args.split]
    out_features_dir = os.path.join(args.output_dir, f'{args.split}-features-obj')
    if args.split == 'custom':
        out_features_dir = os.path.join(args.output_dir, 'valid-features-obj')
    out_metadata_file = os.path.join(out_features_dir, 'metadata.csv')

    os.makedirs(out_features_dir, exist_ok=True)

    with open(out_metadata_file, 'w') as fo:
        writer = csv.DictWriter(fo, fieldnames=FIELDNAMES_OUT, extrasaction='ignore')
        writer.writeheader()

        for in_features_file in in_features_files:
            with open(in_features_file, 'r') as fi:
                reader = csv.DictReader(fi, delimiter='\t', fieldnames=FIELDNAMES_IN)
                for item in tqdm.tqdm(reader):
                    item['image_id'] = int(item['image_id'])
                    item['image_h'] = int(item['image_h'])
                    item['image_w'] = int(item['image_w'])
                    item['num_boxes'] = int(item['num_boxes'])
                    for field in ['boxes', 'features']:
                        item[field] = np.frombuffer(base64.decodebytes(item[field].encode()), dtype=np.float32).reshape(item['num_boxes'], -1)

                    # write features metadata
                    # TODO: include bounding boxes
                    writer.writerow(item)

                    # write features, one file per image
                    np.save(os.path.join(out_features_dir, f"{item['image_id']}.npy"), item['features'])

                    # write bounding boxes, one file per image
                    np.save(os.path.join(out_features_dir, f"{item['image_id']}-boxes.npy"), item['boxes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Object features pre-processing.')

    parser.add_argument('--features-dir',
                        help='Object features data directory.')
    parser.add_argument('--split', choices=['train', 'valid', 'test', 'custom'],
                        help="Data split ('train', 'valid', 'test' or 'custom').")
    parser.add_argument('--custom-feature-dir', default=None,
                        help='Custom feature directory.')
    parser.add_argument('--output-dir', default='output',
                        help='Output directory.')
    parser.add_argument('--device', default='cuda', type=torch.device,
                        help="Device to use ('cpu', 'cuda', ...).")

    main(parser.parse_args())
