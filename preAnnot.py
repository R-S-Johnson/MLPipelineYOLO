"""
preAnnot.py
Description: Takes given incommplete annotations
and uploads them to an existing Label Studio server
for users to validate in a labeling campaign.
Author: Riley Johnson
Last Modified: 5/1/24
"""

from label_studio_sdk import Client
from PIL import Image
from random import randint
import argparse
import json
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description=("Takes given incommplete annotations and uploads " + 
                                                  "them to an existing Label Studio server for users " + 
                                                  "to validate in a labeling campaign."))
    parser.add_argument('-host', '--label-studio-hostname', type=str,
                        required=True, help='url to Label Studio server')
    parser.add_argument('-pid', '--project-id', type=int,
                        default=1, help='Project id from label studio server url (refer to README)')
    parser.add_argument('-t', '--token', type=str,
                        required=True, help='API token from label studio server')

    # Extract arguments from parser
    args = parser.parse_args()
    url = args.label_studio_hostname
    project_id = args.project_id
    token = args.token

    # Initialize Label Studio client
    lsClient = Client(url=url, api_key=token)

    # Check if connection to Label Studio API is successful
    if not lsClient.check_connection():
        print("Failed to connect to Label Studio API.")
        return

    # Retrieve the specified project
    project = lsClient.get_project(project_id)

    # Check if project is found
    if not project:
        print(f"Project {project_id} not found.")
        return

    # Process tasks in the project
    label_paths = os.listdir("prelabels")
    process_tasks(project, project_id, label_paths)

def process_tasks(project, project_id, label_paths):
    # Iterate through each task (image) in the project
    for task in project.get_tasks():
        task_id = task['id']
        image_filename = task['data']['image'][task['data']['image'].find('-')+1:]
        
        # Check if pre-labeled JSON file exists for the image
        if f"{os.path.splitext(image_filename)[0]}.json" in label_paths:
            # Load pre-labeled annotations from JSON file
            with open(f"prelabels/{os.path.splitext(image_filename)[0]}.json") as f:
                labels = json.load(f)

            # Open corresponding image to get its dimensions
            with Image.open(f"YOLOdata/images/{image_filename}") as img:
                img_width, img_height = img.size

            # Generate annotations based on pre-labeled data
            postBody = create_annotations(labels['annotations'], img_width, img_height)
            postBody['project'] = project_id
            postBody['task'] = task_id

            # Create annotations in Label Studio project
            project.create_annotation(task_id, **postBody)

def create_annotations(labels, img_width, img_height):
    annotations = []

    # Iterate through each label in the pre-labeled data
    for label in labels:
        # Create a new annotation object
        tmpAnnot = {
            "id": str(randint(1, 10000)),
            "from_name": "label",
            "to_name": "image",
            "type": "rectanglelabels",
            "origin": "manual",
            "value": {
                "x": label['center']['coordinates'][0] / img_width * 100,
                "y": label['center']['coordinates'][1] / img_height * 100,
                "width": 50 / img_width * 100,
                "height": 50 / img_height * 100,
                "rectanglelabels": [label['label']]
            },
            "original_width": img_width,
            "original_height": img_height
        }
        annotations.append(tmpAnnot)

    # Build body of arguments for API call
    postBody = {
        "created_ago": "0 minutes",
        "completed_by": 1,
        "result": annotations,
        "was_cancelled": False,
        "ground_truth": False,
        "draft_created_at": None,
        "lead_time": None,
        "import_id": None,
        "last_action": None,
        "task": 0,
        "project": 0,
        "updated_by": 1,
        "parent_prediction": None,
        "parent_annotation": None,
        "last_created_by": None
    }
    return postBody

if __name__ == "__main__":
    main()