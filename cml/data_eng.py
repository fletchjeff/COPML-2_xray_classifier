!hdfs dfs -copyFromLocal -p -f dataset/train $STORAGE/datalake/data/xray/train
!hdfs dfs -copyFromLocal -p -f dataset/test $STORAGE/datalake/data/xray/test



#import glob
#from sklearn.model_selection import train_test_split
#import tensorflow as tf
#
#normal_file = glob.glob("chest_xray/normal/*.jpeg")
#pneumonia_file = glob.glob("chest_xray/pneumonia/*.jpeg")
#
#all_files = normal_file + pneumonia_file
#
#
#import random
#
#print(random.choice(foo))
#
#
#
#x_train, x_test = train_test_split(normal_file, test_size=0.2)
#
#
#import pathlib
#data_dir = pathlib.Path("/home/cdsw/chest_xray/normal")
#
#batch_size = 32
#img_height = 224
#img_width = 224
#IMG_SIZE = (224, 224)
#
#train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
#  "/home/cdsw/chest_xray_data/normal",
#  #validation_split=0.2,
#  labels = [1] * 1341,
#  #class_names = ['normal'],
#  #subset="training",
#  seed=123,
#  image_size=(img_height, img_width),
#  batch_size=batch_size)