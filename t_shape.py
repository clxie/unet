from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np 
import os
import glob
import skimage.io as io
import skimage.transform as trans
import shutil
from data import *
from model import *

model = unet(pretrained_weights = './unet_contour_extraction.hdf5',input_size=(480,480,1))

testgene = testGenerator2('/mnt/pc-hf197/chunleixie/ml/mnist/all/test/imgs')

results = model.predict_generator(testgene,41,verbose=1)

saveResult2('data/membrane/testout','/mnt/pc-hf197/chunleixie/ml/mnist/all/test/imgs',results)
