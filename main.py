from model import *
from data import *
from PIL import Image
import keras 
import time
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


data_gen_args = dict(rotation_range=0.0,
                    width_shift_range=0.0,
                    height_shift_range=0.0,
                    shear_range=0.0,
                    zoom_range=0.0,
                    horizontal_flip=True,
                    fill_mode='nearest')

t1 = time.time()
myGene = trainGenerator(2,'/mnt/pc-hf198/chunleixie/ml/mnist/all/train/','imgs_unet','label_unet',data_gen_args)
t2 = time.time()
print('########## prepare data time:',(t2 - t1))
sys.stdout.flush()

model = unet()
model_checkpoint = ModelCheckpoint('unet_contour_extraction.hdf5', monitor='loss',verbose=1, save_best_only=True)
#tensorboard = keras.callbacks.TensorBoard(log_dir='./log')
model.fit_generator(myGene,steps_per_epoch=300,epochs=100,callbacks=[model_checkpoint])
t1 = time.time()
print('########## train model time:',(t1 - t2))
sys.stdout.flush()

#testGene = testGenerator("/mnt/pc-hf198/chunleixie/all/test/imgs")
#results = model.predict_generator(testGene,30,verbose=1)
#saveResult("data/membrane/imgs_out","data/membrane/imgs",results,flag_multi_class=false)
