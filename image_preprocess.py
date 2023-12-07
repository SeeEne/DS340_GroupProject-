import os
import cv2

def new_image_name(filename, prefix):
    name, ext = os.path.splitext(filename)
    return f"{prefix}{name}{ext}"

def resize_images_by_name(image_names, root_folder, output_folder, target_size=(64, 64)):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    count = 0
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".jpg") and filename in image_names:
                img = cv2.imread(os.path.join(dirpath, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img, target_size, interpolation = cv2.INTER_AREA)
                #mg_resized = img.resize(target_size, Image.ANTIALIAS)
                output_name = new_image_name(filename, count)
                cv2.imwrite(os.path.join(output_folder, output_name), img_resized)
            count += 1

image_names = ["0.jpg","1.jpg","2.jpg","3.jpg","4.jpg","5.jpg"]  # Replace with the names of the images you want to select

root_folder = "data_autoencoder/raw_images"  # Replace with the path to the root folder containing JPG images and subfolders
output_folder = "data_autoencoder/process_images"  # Replace with the path where you want to save resized JPG images

resize_images_by_name(image_names, root_folder, output_folder, target_size=(64, 64))