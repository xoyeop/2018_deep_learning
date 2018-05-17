import os
import numpy as np
import tensorflow as tf

import image_preprocessing_util as iputil
import util as myutil

# from PIL import Image

import pprint
pp = pprint.PrettyPrinter()

from scipy.misc import imsave

TRAIN  = "train"
TEST  = "test"

flags = tf.app.flags
flags.DEFINE_string("image_dir", "Images", "The directory of cropped and resized dog images [Images]")
flags.DEFINE_string("output_dir", "tfrecords", "The directory of tfrecord_output [tfrecords]")
flags.DEFINE_boolean("cropping", "True", "Boolean variable about cropped dog faces [True]")
flags.DEFINE_float("test_ratio", "0.2", "The ratio of test image data set [0.8]")
flags.DEFINE_integer("img_width", 32, "Width of image")
flags.DEFINE_integer("img_height", 32, "Height of image")

FLAGS = flags.FLAGS

def get_total_data():
    """
        get_total_data()
        - Get all images in image saved folder and make numpy array
        - Image folder format: Images/{breed}/{filename}
    """
    IMAGE_DIR = FLAGS.image_dir

    breeds = os.listdir(IMAGE_DIR)
    image_set = {}

    for breed in breeds:
        image_dir = os.path.join(IMAGE_DIR, breed)
        image_set[breed] = os.listdir(image_dir)

        total_data = []
        for breed in image_set.keys():
            total_data.extend([ [filename,breed] for filename in image_set[breed]])
        total_data = np.array(total_data)
    return total_data

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _get_target_dir():
    if FLAGS.cropping:
        TARGET_DIR = "cropping"
    else:
        TARGET_DIR = "general"

    return TARGET_DIR

def get_splitted_data(total_data):
    """
    Returns:
    """

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
         total_data[:,0], total_data[:,1], test_size=FLAGS.test_ratio, random_state=1)
    return X_train, X_test, y_train, y_test

def generate_patches():
    with open('testfile.txt', 'r') as f:
        for patch in f.readlines():
            yield patch[:-1]

def persistence_image_data_to_tfrecords(
    x_data, y_data, data_type, split_index=128):
    """
        persistence_image_data_to_tfrecords(x_data, y_data, data_type, split_index)
        - Make images into tfrecords and save
        - Crop / Resize / Convert rgb into grayscale
    """

    TARGET_DIR = _get_target_dir()
    OUTPUT_DIR = os.path.join(FLAGS.output_dir,TARGET_DIR,data_type)
    IMAGE_DIR = FLAGS.image_dir

    myutil.check_directory("./resized_img")   # Check directory of "resized_img"
    myutil.check_directory("./tfrecords")
    myutil.check_directory("./tfrecords/cropping")
    myutil.check_directory("./tfrecords/cropping/train")
    myutil.check_directory("./tfrecords/cropping/test")
    myutil.check_directory(OUTPUT_DIR)    # Check directory of OUTPUT_DIR

    writer = None
    sess = None
    current_index = 0

    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(y_data)
    y_data_size = len(le.classes_)

    # https://stackoverflow.com/questions/45427637/is-there-a-more-simple-way-to-handle-batch-inputs-from-tfrecords

    for images_filename, y_label in zip(x_data, y_data):
        if not(images_filename[-3:] == "jpg"):
            print("Error - ", images_filename)
            continue
        if current_index % split_index == 0:
            if writer:
                writer.close()
            if sess:
                sess.close()
            tf.reset_default_graph()
            graph = tf.get_default_graph()
            sess = tf.Session(graph=graph)
            sess.run(tf.global_variables_initializer())

            record_filename = "{output_dir}/{data_type}-{current_index}.tfrecords".format(
                output_dir=OUTPUT_DIR, data_type=data_type, current_index=current_index
            )
            print("=============>" , record_filename)
            print("current index : {0}".format(current_index,))
            writer = tf.python_io.TFRecordWriter(record_filename)

        file_full_path = os.path.join(
                IMAGE_DIR, y_label,  images_filename)
        try:
            image_file = tf.read_file(file_full_path)
            decoded_img = tf.image.decode_jpeg(image_file, channels=3)   # 'jpg' file decoding

        except tf.errors.InvalidArgumentError as e:
            print(e)
            print("Error : ", images_filename)
            continue

        image_list = [decoded_img]

        # If I wanna crop images
        if FLAGS.cropping:
            image_list = []

            # Get size information
            size_info = iputil.get_orginal_size_info(y_label, images_filename)
            target_img = tf.image.resize_images(decoded_img, [size_info["height"], size_info["width"]])
            # Get bounding box to crop image (it is list)
            # bounding_box = [[xmin,ymin,bounding_width,bounding_height]]
            bounding_box = iputil.get_bounding_size_info(y_label, images_filename)

            # Crop image with bounding box
            for box in bounding_box:

                # tf.img.crop_to_bounding_box
                # tf.img.* 함수들은 width보다 height를 먼저 쓴다
                # -> 이미지 좌표상 x,y,width,height의 순서를 잘 확인하지 않으면 전처리가 잘못 적용되니 유의할 것!
                cropped_img = tf.image.crop_to_bounding_box(
                    target_img,
                    box[1], box[0], box[3], box[2]
                )

                # cropped_img = tf.image.crop_and_resize(decoded_img, bounding_box, i, crop_size=[FLAGS.img_height, FLAGS.img_width])
                """
                tf.image.crop_and_resize를 사용하고 싶은데 뭔가 자꾸 오류가 남...
                tf.image.crop_and_resize 함수 내에 세팅하는 값의 문제 - 함수 사용법을 좀더 살펴봐야 함!
                tf.image.crop_and_resize(
                    image,
                    boxes,
                    box_ind,
                    crop_size,
                    method='bilinear',
                    extrapolation_value=0,
                    name=None
                )
                """

                image_list.append(cropped_img)

        for image in image_list:
            try:
                imsave(
                    "./resized_img/" + images_filename + "_" + str(current_index) + ".jpeg",
                    sess.run(image)
                )
            except OSError as e:
                print(e)

            gray_img = tf.image.rgb_to_grayscale(image) # rgb to grayscale
            resized_img = tf.image.resize_images(gray_img, [FLAGS.img_width, FLAGS.img_height]) # Resize image with new width and height

            image_bytes = sess.run(tf.cast(resized_img, tf.uint8)).tobytes()

            y_data_label = le.transform([y_label])
            lbl_one_hot = tf.one_hot(y_data_label[0], y_data_size, 1, 0)
            image_label = sess.run(tf.cast(lbl_one_hot, tf.uint8)).tobytes()


            feature = {'label': _bytes_feature(image_label),
                        'image': _bytes_feature(image_bytes)}

            example = tf.train.Example(
                    features = tf.train.Features(
                                        feature=feature))

            writer.write(example.SerializeToString())
            current_index += 1
            
    writer.close()

def main(_):
    print ('Converting JPG to tfrecord datatype')
    print ('Argument setup')
    pp.pprint(flags.FLAGS.__flags)
    print ('---------------------------------')

    total_data = get_total_data()
    number_of_data_types = len(np.unique(total_data[:, 1]))
    print("The number of data : {0}".format(total_data.shape[0],))
    print("The number of breeds : {0}".format(number_of_data_types,))

    print('---------------------------------')
    X_train, X_test, y_train, y_test = get_splitted_data(total_data)
    print("Train / Test ratio : {0:.2f} / {1:.2f}".format( 1-FLAGS.test_ratio, FLAGS.test_ratio ))
    print("Number of train data set : {0}".format(len(X_train)))
    print("Number of test data set : {0}".format(len(X_test)))


    print('---------------------------------')

    persistence_image_data_to_tfrecords(X_train, y_train, data_type=TRAIN, split_index=128)
    persistence_image_data_to_tfrecords(X_test, y_test, data_type=TEST, split_index=128)


if __name__ =="__main__":
    tf.app.run()
