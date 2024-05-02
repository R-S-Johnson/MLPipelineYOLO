"""
labelPostProc.py
Description: Translates exported labels from the completed
Label Studio labeling campaign to a format ready for YOLOv8
OBB training.
Author: Riley Johnson
Last Modified: 5/1/24
"""

import json
import numpy as np
import argparse
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description=("Translates exported labels from the completed " +
                                                  "Label Studio labeling campaign to a format ready " +
                                                  "for YOLOv8 OBB training."))
    parser.add_argument('-annot', '--annotations-file', type=str,
                        required=True, help='Exported annotations json from Label Studio')
    parser.add_argument('-class', '--classes-file', type=str,
                        required=True, help='Json classes mapped to indexes (refer to README)')

    # Extract arguments from parser
    args = parser.parse_args()
    annotations_file = args.annotations_file
    classes_file = args.classes_file

    # process data in annotations file
    process_data(annotations_file, classes_file)

def rotated_rect_to_corners(x_center, y_center, width, height, rotation):
    # Convert rotation angle from degrees to radians
    rotation_rad = np.radians(rotation)

    # Half width and half height
    w2 = width / 2
    h2 = height / 2

    # Coordinates of the corners relative to the center of the rectangle
    corners = np.array([
        [-w2, -h2],
        [w2, -h2],
        [w2, h2],
        [-w2, h2]
    ])

    # Rotation matrix for rotation around the origin
    rotation_matrix = np.array([
        [np.cos(rotation_rad), -np.sin(rotation_rad)],
        [np.sin(rotation_rad), np.cos(rotation_rad)]
    ])

    # Rotate the corners around the origin (0, 0) and translate to the center (x_center, y_center)
    rotated_corners = np.dot(corners, rotation_matrix.T) + np.array([x_center, y_center])
    return rotated_corners

def process_data(annotations_file, classes_file):
    # Load annotations and classes data
    with open(annotations_file) as f:
        data = json.load(f)
    with open(classes_file) as f:
        classes = json.load(f)

    # Create YOLO data directory structure if not already presen
    if not os.path.exists("YOLOdata"):
        os.mkdir("YOLOdata")
        os.mkdir("YOLOdata/labels")
        os.mkdir("YOLOdata/images")
        print("ERROR: create YOLO data directory")
        exit()
    else:
        if not os.path.exists("YOLOdata/labels"):
            os.mkdir("YOLOdata/labels")
        if not os.path.exists("YOLOdata/images"):
            print("ERROR: images folder not found in YOLO data directory")
            os.mkdir("YOLOdata/images")

    # Grab necessary data slightly reorganized
    tmp = [{
        'fileName': image['file_upload'],
        'annotations': [{
            'className': label['value']['rectanglelabels'][0],
            'x_center': label['value']['x'],
            'y_center': label['value']['y'],
            'width': label['value']['width'],
            'height': label['value']['height'],
            'rotDeg': label['value']['rotation']
        } for label in image['annotations'][0]['result']],
        'imsize': [image['annotations'][0]['result'][0]['original_width'],
                image['annotations'][0]['result'][0]['original_height']]
    } for image in data]

    # Write YOLO labels for each image
    for image in tmp:
        imageName = image['fileName'].split('.')[0][image['fileName'].split('.')[0].find('-') + 1:]
        
        with open(f"YOLOdata/labels/{imageName}.txt", 'w') as f:
            for obbLabel in image['annotations']:
                # Translate xywhr to 4 points
                corners = rotated_rect_to_corners(obbLabel['x_center'], obbLabel['y_center'],
                                                  obbLabel['width'], obbLabel['height'],
                                                  obbLabel['rotDeg'])

                # reformat function output for file writing
                corners = [str(pnt) for pnt in corners.flatten().tolist()]

                # Write label in YOLO format
                class_id = str(classes[obbLabel['className']])
                f.write(' '.join([class_id] + corners) + '\n')

if __name__ == '__main__':
    main()