import cv2
import pytesseract
import os
import matplotlib.pyplot as plt

from src.combine_jsons import *
from src.timer_task import *
from src.is_timer import *

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # TODO: Change this to the Tessaract path on the system


if __name__ == "__main__":
    # Specify the folder path containing frames
    frames_root_folder = "/cs-share/pradalier/tmp/judo/frames/"

    json_folder = "json/"

    # Specify the region of interest as (x1, y1, x2, y2)
    roi_coordinates_timer = (660, 625, 750, 665)

    # Get the list of folders containing frames
    frame_mat_folders = [os.path.join(frames_root_folder, folder) for folder in os.listdir(frames_root_folder) if os.path.isdir(os.path.join(frames_root_folder, folder))]
    frame_folders = [os.path.join(frame_mat_folder, folder) for frame_mat_folder in frame_mat_folders for folder in os.listdir(frame_mat_folder) if os.path.isdir(os.path.join(frame_mat_folder, folder))]

    # Extract the timer from the frames
    print("-------------------")
    print("Extracting timer...")
    print("-------------------")

    parallelize_extraction(frame_folders, json_folder, roi_coordinates_timer)

    print("-------------------")
    print("Extraction done!")
    print("-------------------")

    # Combine the JSON files
    print("-------------------")
    print("Combining JSON files...")
    print("-------------------")

    folder_path = "json/"
    output_file = "combined.json"

    combine_json_files(json_folder, "combined.json")
    print("-------------------")
    print("Combining done!")
    print("-------------------")