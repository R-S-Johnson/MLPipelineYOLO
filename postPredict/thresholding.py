"""
Thresholding.py
Tools for sorting out the best n
detections for each class in each
image predictions made by YOLOv8 OBB.
Author: Riley Johnson
Last Modified: 5/1/24
"""

import pandas as pd
import argparse
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description=("Tools for sorting out the best n detections for each class " +
                                                  "in each image predictions made by YOLOv8 OBB."))
    parser.add_argument('-labels', '--labels-directory', type=str,
                        required=True, help='Directory with YOLO prediction txt file outputs')
    parser.add_argument('-thresh', '--threshold', type=int,
                        default=10, help='Threshold n for number of best detections to look for')
    parser.add_argument('-tot', '--total-classes', type=int,
                        required=True, help='Total number of detection classes')
    
    # Extract argumets from parser
    args = parser.parse_args()
    labels_dir = args.labels_directory
    threshold = args.threshold
    total_classes = args.total_classes

    # Process labels in the labels directory
    process_labels(labels_dir, threshold, total_classes)


def process_labels(labels_dir, threshold, total_classes):

    # Define column names for the DataFrame: class, 4 (x, y) coordinates, and the confidence
    columns = ["class"] + [str(i) for i in range(1,9)] + ["conf"]

    # For all files in the labels directory
    for path in os.listdir(labels_dir):
        # Read and parse the predictions file
        with open(os.path.join(labels_dir, path)) as f:
            predictions = pd.DataFrame([line.strip().split(' ') for line in f.readlines()],
                                    dtype=str,
                                    columns=columns)
        replace_lines = []

        # Process predictions for each class
        for class_i in range(total_classes + 1):
            filt = predictions[predictions["class"] == str(class_i)]
            if len(filt) > threshold:
                replace_lines += filt.sort_values(by='conf', ascending=False).head(threshold).values.tolist()
            elif 0 < len(filt) <= threshold:
                replace_lines += filt.values.tolist()

        # Write updated predictions back to the file
        to_write = [' '.join(line) + "\n" for line in replace_lines]
        with open(f"{labels_dir}/{path}", 'w') as f:
            f.writelines(to_write)


if __name__ == '__main__':
    main()