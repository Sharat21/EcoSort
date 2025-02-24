#Runs all code in setup.py. This creates the file structure and dataset


#Create dataset
#Parameters are set in create_dataset.py
from create_dataset_2 import create_dataset
create_dataset()


# #Runs all code in image_seg.py -> Segments all images in ./inputs and saves all objects in ./objects. Images are saved in ./classifications, as original_image_name_label.jpg for training images and original_image_name.jpg for predictions  
# import image_seg

# image_seg.segment_images_training()
