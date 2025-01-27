import argparse
import os
import torch
import tqdm
import numpy as np

from torchvision import transforms

import sys
sys.path.insert(0,os.getcwd())

from data import ImageDataset, read_split_image_ids_and_paths
from model.inception import inception_v3_base


def main(args):
    if args.split != 'custom':
        image_ids, image_paths = read_split_image_ids_and_paths(args.split)
        image_paths = [os.path.join(args.ms_coco_dir, 'images', image_path) for image_path in image_paths]
        features_dir = os.path.join(args.output_dir, f'{args.split}-features-grid')
    if args.split == 'custom':
        image_paths_base = os.listdir(os.path.join(args.ms_coco_dir, 'images/val2014'))
        image_ids = [os.path.splitext(image_base)[0][-6] for image_base in image_paths_base]
        image_paths = [os.path.join(args.ms_coco_dir, 'images/val2014', image_base) for image_base in image_paths_base]
        features_dir = os.path.join(args.output_dir, 'valid-features-grid')

    os.makedirs(features_dir, exist_ok=True)

    inception = inception_v3_base(pretrained=True)
    inception.eval()
    inception.to(args.device)

    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(image_ids, image_paths, transform=transform)
    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=args.batch_size,
                                         num_workers=args.num_workers,
                                         pin_memory=args.device.type == 'cuda',
                                         shuffle=False)

    with torch.no_grad():
        for imgs, ids in tqdm.tqdm(loader):
            outs = inception(imgs.to(args.device)).permute(0, 2, 3, 1).view(-1, 64, 2048)
            for out, id in zip(outs, ids):
                out = out.cpu().numpy()
                if type(id) != str:
                    id = str(id.item())
                np.save(os.path.join(features_dir, id), out)
    print('Saved grid features from inceptionv3 to %s' % features_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MS-COCO captions pre-processing.')

    parser.add_argument('--ms-coco-dir',
                        help='MS-COCO data directory.')
    parser.add_argument('--split', choices=['train', 'valid', 'test', 'custom'],
                        help="Data split ('train', 'valid', 'test', or 'custom').")
    parser.add_argument('--output-dir', default='output',
                        help='Output directory.')
    parser.add_argument('--device', default='cuda', type=torch.device,
                        help="Device to use ('cpu', 'cuda', ...).")
    parser.add_argument('--batch-size', default=8, type=int,
                        help="Image batch size.")
    parser.add_argument('--num-workers', default=0, type=int,
                        help="Number of data loader workers.")

    main(parser.parse_args())
