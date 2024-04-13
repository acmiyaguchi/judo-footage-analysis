import json
import os
import shutil

import requests
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Function to parse Label Studio JSON file
def parse_label_studio_json(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    # annotations = data['annotations']
    """
    dict_keys(['id', 'annotations', 'file_upload', 'drafts', 'predictions', 'data', 'meta', 'created_at', /
    'updated_at', 'inner_id', 'total_annotations', 'cancelled_annotations', 'total_predictions', 'comment_count', /
    'unresolved_comment_count', 'last_comment_updated_at', 'project', 'updated_by', 'comment_authors'])
    """
    dataset = []
    # for annotation in annotations:
    for d in data:
        image_path = d["data"]["image"]
        labels = d["annotations"][0]["result"]
        try:
            labels = labels[0]["value"]["choices"][0]
        except IndexError:
            continue
        dataset.append((image_path, labels))
    return dataset


# download images from labelstudio
def download_image(url, output_dir):
    filename = os.path.join(output_dir, os.path.basename(url))
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    return filename


# Function to organize the dataset
def organize_dataset(dataset, output_dir):
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    train_set, test_set = train_test_split(dataset, test_size=0.4, random_state=42)

    organize_split(train_set, train_dir)
    organize_split(test_set, test_dir)


def organize_split(dataset, output_dir):
    for image_path, label in tqdm(dataset):
        label_dir = os.path.join(output_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        download_image(image_path, label_dir)


# Path to Label Studio JSON file
json_file = "/home/GTL/tsutar/intro_to_res/referee.json"

# Output directory for the YOLOv5 dataset
output_dir = "/home/GTL/tsutar/intro_to_res/referee_dataset_v2"

# Parse Label Studio JSON file
dataset = parse_label_studio_json(json_file)

# Organize the dataset
organize_dataset(dataset, output_dir)
