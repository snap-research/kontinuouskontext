import cv2
import numpy as np
from PIL import Image 

# this will be used to process the output image 
def process_image(img):
    """
    Crops the input PIL image to the largest possible region with a 3:4 (width:height) aspect ratio,
    centered in the image.
    """
    target_ratio = 3 / 4  # width / height
    width, height = img.size
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Image is too wide, crop width
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        right = left + new_width
        top = 0
        bottom = height
    else:
        # Image is too tall, crop height
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        bottom = top + new_height
        left = 0
        right = width

    return img.crop((left, top, right, bottom))


# Implement sanitize_filename to process the prompt-based name for saving as a filename
def sanitize_filename(filename):
    # Remove or replace characters that are invalid in filenames
    import re
    # Remove any character that is not alphanumeric, space, dot, underscore, or hyphen
    filename = re.sub(r'[^\w\s\.-]', '', filename)
    # Replace spaces with underscores
    filename = filename.replace(' ', '_')
    # Optionally, truncate to a reasonable length
    return filename[:300]
