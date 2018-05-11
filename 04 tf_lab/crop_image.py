import os
import numpy as np
import tensorflow as tf

import image_preprocessing_util as iputil

from PIL import Image

import pprint
pp = pprint.PrettyPrinter()

from scipy.misc import imsave

flags = tf.app.flags
flags.DEFINE_string("image_dir", "Images", "The directory of dog images [Images]")
flags.DEFINE_string("output_dir", "Cropped", "The directory of cropped output [Cropped]")
flags.DEFINE_integer("img_size", 32, "Image size")

FLAGS = flags.FLAGS

def save_crop_image():
    IMAGE_DIR = FLAGS.image_dir
    OUTPUT_DIR = FLAGS.output_dir

    if not(os.path.exists(OUTPUT_DIR)):
        os.makedirs(OUTPUT_DIR)
        print("Directory create : {0}".format(OUTPUT_DIR,))

    bleeds = os.listdir(IMAGE_DIR)
    image_set = {}

    for bleed in bleeds:

        image_dir = os.path.join(IMAGE_DIR, bleed)
        image_set[bleed] = os.listdir(image_dir)

        #total_data = []
        for bleed in image_set.keys():
            check_dir = os.path.join(OUTPUT_DIR, bleed)

            if(os.path.exists(check_dir)):
                continue

            for filename in image_set[bleed]:

                img_path = os.path.join(IMAGE_DIR, bleed, filename)
                image = Image.open(img_path)

                # Get bounding box to crop image [xmin, ymin, bounding_width, bounding_height]
                bounding_box = iputil.get_bounding_size_info(bleed, filename)

                left = bounding_box[0][0]
                top = bounding_box[0][1]
                width = bounding_box[0][2]
                height = bounding_box[0][3]

                box = (left, top, left + width, top + height)

                # Crop and resize image
                crop_img = image.crop(box)
                img = crop_img.resize((FLAGS.img_size, FLAGS.img_size), Image.ANTIALIAS)

                crop_dir = os.path.join(OUTPUT_DIR, bleed)

                if not(os.path.exists(crop_dir)):
                    os.makedirs(crop_dir)
                    print("Directory create : {0}".format(crop_dir,))

                rgb_img = img.convert('RGB')

                file_full_path = os.path.join(crop_dir,filename)
                rgb_img.save(file_full_path)

def main(_):
    print ('Save cropped image')
    print ('Argument setup')
    pp.pprint(flags.FLAGS.__flags)
    print ('---------------------------------')

    save_crop_image()

if __name__ =="__main__":
    tf.app.run()
