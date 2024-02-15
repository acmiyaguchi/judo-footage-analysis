import cv2
import pytesseract
import os
import matplotlib.pyplot as plt
import re
import json
from concurrent.futures import ThreadPoolExecutor

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # TODO: Change this to the Tessaract path on the system

def is_timer(input_string):
    # Remove leading and trailing whitespaces
    input_string = input_string.strip()

    # Extract the first line
    input_string = input_string.split('\n')[0].strip()

    # Define the regular expression pattern for the timer format
    pattern = re.compile(r'^\d{1,2}\s*:\s*\d{2}$')

    # Check if the input string matches the pattern
    if not pattern.match(input_string):
        return (False, None, None)

    # Split the string into minutes and seconds
    minutes, seconds = map(int, input_string.split(':'))

    # print(f"minutes: {minutes}, seconds: {seconds}")

    # Check if seconds are within the valid range (0 to 59)
    if 0 <= seconds <= 59:
        return (True, minutes, seconds)
    else:
        return (False, None, None)

def extract_timer_from_folder(folder_path, json_path, roi_coordinates):

    # Extract last folder in folder_path
    last_folder_path = folder_path.split('/')[-1]
    mat_folder_path = folder_path.split('/')[-2]

    # Create the json file to store the results if it doesn't exist
    json_filename = os.path.join(json_path, 'timer_' + mat_folder_path + '_' + last_folder_path + '.json')
    print(f"json_filename: {json_filename}")
    if not os.path.exists(json_filename):
        with open(json_filename, 'w') as f:
            f.write("[]")

    results = []

    for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                # print(f"Treating {filename}...")
                image_path = os.path.join(folder_path, filename)
                # Read the image
                img = cv2.imread(image_path)

                # Extract the specified region of interest
                roi = img[roi_coordinates[1]:roi_coordinates[3], roi_coordinates[0]:roi_coordinates[2]]

                # Convert the region of interest to grayscale for better OCR accuracy
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Apply thresholding if needed
                _, thresh = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Use Tesseract to extract text from the region of interest
                text = pytesseract.image_to_string(thresh)

                result, minutes, seconds = is_timer(text)

                results.append({'filename': folder_path + filename, 'available': result, 'minutes': minutes, 'seconds': seconds, 'raw_text': text})
    
    # Write the results to the json file
    with open(json_filename, 'w') as f:
        json.dump(results, f)
    
    print(f"Extraction done for {folder_path}!")

def parallelize_extraction(frame_folders, json_folder, roi_coordinates):
    with ThreadPoolExecutor(max_workers=None) as executor:
        futures = []
        for frame_folder in frame_folders:
            futures.append(executor.submit(extract_timer_from_folder, frame_folder, json_folder, roi_coordinates))
        
        # Wait for all tasks to complete
        for future in futures:
            future.result()