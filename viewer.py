import os
import sys
import csv
import json
import shutil
import pandas as pd

from PIL import Image, ImageDraw

# MS-COCO directory (must be a relative path)
ms_coco_dir = 'ms-coco'

# File containing generated captions
predictions_file = 'demo/demo-predictions.json'


def image_path(coco_dir, image_id):
    return os.path.join(coco_dir, 'images', 'val2014', f'COCO_val2014_{image_id:012d}.jpg')


def image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


def image_html(image_path, scale):
    width, height = image_size(image_path)
    return f'<img src="{image_path}" width="{width * scale}" height="{height * scale}"/>'


def result_html(coco_dir,
                image_ids,
                image_captions,
                image_scale=0.3):

    image_paths = [image_path(coco_dir, image_id) for image_id in image_ids]

    df = pd.DataFrame(dict({'ImageID': image_ids,
                            'Image': image_paths,
                            'Caption': image_captions}))

    if not os.path.isdir("demo/demo_outputs"):
        os.mkdir("demo/demo_outputs")
    for i in range(len(image_paths)):
        my_image = Image.open(image_paths[i])
        image_editable = ImageDraw.Draw(my_image)
        image_editable.rectangle(((0, 00), (500, 30)), fill="black")
        image_editable.text((15,15), image_captions[i], (237, 230, 211))
        my_image.save("demo/demo_outputs/output_"+os.path.basename(image_paths[i]))

    print("Images saved to demo/demo_outputs")

def viewer(predictions_file):
    image_ids = []
    image_captions = []

    with open(predictions_file) as f:
        preds = json.load(f)
    
        for pred in preds:
            image_ids.append(pred['image_id'])
            image_captions.append(pred['caption'])    
    
    result_html(ms_coco_dir, image_ids, image_captions)

def see_test_ids_file():
    with open("output/test-ids.txt") as f:
        f = f.read().split("\n")
        print(f[0], len(f))

def see_features_file():
    import base64
    import numpy as np
    csv.field_size_limit(sys.maxsize)
    with open("ms-coco/features_original/karpathy_val_resnet101_faster_rcnn_genome.tsv") as f:
        features_file = list(csv.reader(f, delimiter="\t"))
        print(features_file[0][:4])
        boxes = np.frombuffer(base64.decodebytes(bytes(features_file[0][4], 'utf-8')))
        features = np.frombuffer(base64.decodebytes(bytes(features_file[0][5], 'utf-8')))
        print("BOXES\n", boxes, boxes.shape)
        print("FEATURES\n", features, features.shape)
        
def convert_xywh_to_xyxy(bbox):
    return [bbox[0],bbox[1],bbox[2]+bbox[0],bbox[3]+bbox[1]]
    
def convert_cat_id_to_name(cat_id, categories):
    for x in categories:
        if x['id']==cat_id:
            return x['name']
    print("Invalid category id")
    return None

def generate_dict(annotation, categories):
    cat_name = convert_cat_id_to_name(annotation['category_id'], categories)
    if cat_name:
        return {"rect": convert_xywh_to_xyxy(annotation['bbox']), "class": cat_name}

def see_annotations():
    files = {'captions_file' : "ms-coco/annotations/captions_val2014.json",
            'instances_file' : "ms-coco/annotations/instances_train2014.json",
            'person_keypoints_file' : "ms-coco/annotations/person_keypoints_val2014.json"}
    with open(files['instances_file']) as f:
        content = json.load(f)
        image_id = 550878
        image = [x for x in content['images'] if x['id']==image_id][0]
        print(image['file_name'])
        print(image['coco_url'])
        labels = [generate_dict(x, content['categories']) for x in content['annotations'] if x['image_id']==image_id]
        print("annotations: ", labels)

def find_unprocessed():
    line_list = '/storage/che011/SGB/scene_graph_benchmark/tools/mini_tsv/data/train_original.linelist.tsv'
    instances_file = "ms-coco/annotations/instances_train2014.json"
    output_dir = '/storage/che011/ICT/fairseq-image-captioning/splits/karpathy_custom_images.txt'
    with open(instances_file) as f:
        content = json.load(f)
    with open(line_list) as f: #, open(output_dir, 'w') as output:
        image_ids = [int(x.strip()) for x in f.read().split("\n") if any(char.isdigit() for char in x)]
        unprocessed = [x for x in range(0,113287) if x not in image_ids]
        with open('/storage/che011/SGB/scene_graph_benchmark/tools/mini_tsv/data/missed_train.linelist.tsv','w') as output:
            output.write('\n'.join([str(x) for x in unprocessed]))
        # unprocessed_filenames = []
        # for image_id in unprocessed:
            # try:
                # unprocessed_filenames.append(os.path.join('train2014',[x['file_name'] for x in content['images'] if x['id']==image_id][0]))
            # except IndexError:
                # print(image_id)
        # output.write("\n".join(unprocessed_filenames))

def find_missing_ids():
    import data
    for split in ['train','valid','test']:
        image_ids_file = os.path.join('output', f'{split}-ids.txt')
        image_ids = data.read_image_ids(image_ids_file, non_redundant=False)
    
        features_dir = os.path.join('output', f'{split}-features-obj')
        image_metadata_file = os.path.join(features_dir, 'metadata.csv')
        image_metadata_file = os.path.join(features_dir, 'metadata.csv')
        image_metadata = data.read_image_metadata(image_metadata_file)
        
        print(split, [x for x in image_ids if x not in image_metadata.keys()])
    
        

# viewer(predictions_file)
# see_test_ids_file()
# see_features_file()
# see_annotations()
# find_unprocessed()
find_missing_ids()