"""
load pre-trained keras imagenet network and keras labels
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow
from tensorflow.keras.applications import vgg16, resnet50
from tensorflow.keras.optimizers import SGD
import json



vgg16_model = vgg16.VGG16(weights='imagenet')
vgg16_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy ')

resnet50_model = resnet50.ResNet50(weights='imagenet')
resnet50_model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='sparse_categorical_crossentropy')
#resnet_model = resnet50.ResNet50(weights='imagenet')

#resnet_model.save('D:/tcav-master/model')

CLASS_INDEX = None
CLASS_INDEX_PATH = ('https://s3.amazonaws.com/deep-learning-models/'
                    'image-models/imagenet_class_index.json')

fpath = tensorflow.keras.utils.get_file(
            'imagenet_class_index.json',
            CLASS_INDEX_PATH,
            cache_subdir='models',
            file_hash='c2c37ea517e94d9795004a39431a14cb')
with open(fpath) as f:
    CLASS_INDEX = json.load(f)

labels = CLASS_INDEX.values()
label = [i[1] for i in labels]
with open('imagenet-keras-label.txt', 'w') as filehandle:
    for listitem in label:
        filehandle.write('%s\n' % listitem)


vgg16_model.save('D:/tcav-master/kerasmodel/vgg16.hdf5')
resnet50_model.save('D:/tcav-master/kerasmodel/resnet50.hdf5')


