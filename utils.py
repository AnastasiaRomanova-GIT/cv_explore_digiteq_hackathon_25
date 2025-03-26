import cv2
import os
import time

def create_template_from_labeled_image(row, image_folder):
    """
    Create a template by cropping a labeled emoji from the
    first file in the dataset.
    """
    image_path = os.path.join(image_folder, row['file_name'])
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # get the coordinates of the upper left corner of the emoji
    # from the df
    # add height and width to get the lower right corner
    x, y = row['x'], row['y']
    h, w = 50, 50  #emoji size

    template = image[y:y+h, x:x+w]

    return template, h, w