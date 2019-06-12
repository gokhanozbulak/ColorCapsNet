#!/usr/bin/python

import os
import argparse
import timeit
import cv2
import numpy as np
from scipy.io import loadmat,savemat
from keras import backend as K
from keras.models import Model
from keras.callbacks import CSVLogger,ModelCheckpoint
from keras.applications.vgg19 import VGG19
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Dense, Reshape, Flatten
from capsulelayers import CapsuleLayer,PrimaryCap
from keras.regularizers import l1,l2,l1_l2

K.set_image_data_format('channels_last')

print 'Initializing pre-trained model..'
pre_trained_model = VGG19()

NA = 'not available'

# PARAMETERS
RUN = 1
ROUTINGS = 3
N_CLASS = 10	# complexity
OPTIMIZER = 'adam'
LOSS = 'mse'
EPOCHS = 2
BATCH_SIZE = 128
DATASET = 'ntire'
DATA_PATH = '../data/train_128_128.npz'
PRETRAINED_MODEL_PATH = 'pretrained_model.h5'

# PATHS
RUN_PATH = os.path.join('runs', str(RUN))
MODEL_PATH = os.path.join(RUN_PATH, 'model_'+str(EPOCHS)+'.h5')
MODEL_SUMMARY_PATH = os.path.join(RUN_PATH, 'model_summary.txt')
OUT_PATH = os.path.join(RUN_PATH, 'out')
LOG_PATH = os.path.join(RUN_PATH, 'log.csv')


def load_ntire():
    data = np.load(DATA_PATH)
    x_gray = data['arr_0']
    x_color = data['arr_1']
    x_gray = x_gray.astype('float32') / 255.
    x_color = x_color.astype('float32') / 255.
    return (x_gray,x_color)

def build_model(input_shape):
    # encoder
    x = Input(shape=input_shape)

    conv1 = Conv2D(filters=64, kernel_size=3, padding='same', trainable=True, name='conv1')(x)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    conv1 = Conv2D(filters=64, kernel_size=3, padding='same', trainable=True, name='conv2')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=N_CLASS, dim_capsule=16, routings=ROUTINGS,
                             name='digitcaps')(primarycaps)

    digitcaps = Flatten()(digitcaps)
    decoder = Dense(512, activation='relu', input_dim=16*N_CLASS)(digitcaps)
    decoder = Dense(1024, activation='relu')(decoder)
    decoder = Dense(np.prod(input_shape), activation='sigmoid')(decoder)
    decoder = Reshape(input_shape)(decoder)

    model = Model(x, decoder)

    # transfer weights from first 2 layers of VGG-19
    weights = pre_trained_model.layers[1].get_weights()[0][:,:,:,:]
    #weights = np.reshape(weights, (3,3,1,64))
    bias = pre_trained_model.layers[1].get_weights()[1]
    model.layers[1].set_weights([weights, bias])
    weights = pre_trained_model.layers[2].get_weights()[0]
    bias = pre_trained_model.layers[2].get_weights()[1]
    model.layers[4].set_weights([weights, bias])

    return model

def train(data):
    (x_gray,x_color) = data
    model = build_model(x_color.shape[1:])
    if PRETRAINED_MODEL_PATH != NA:
        model.load_weights(PRETRAINED_MODEL_PATH)
        print 'Pretrained model {0} loaded..'.format(PRETRAINED_MODEL_PATH)
 
    model.compile(optimizer=OPTIMIZER, loss=LOSS)
    
    model.summary()
    with open(MODEL_SUMMARY_PATH, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x+'\n'))
    log = CSVLogger(LOG_PATH)

    checkpoint = ModelCheckpoint('runs/'+str(RUN)+'/weights-{epoch:02d}.h5', monitor='loss', save_best_only=True, save_weights_only=True, verbose=1)

    model.fit(x_gray, x_color,
		epochs=EPOCHS,
		batch_size=BATCH_SIZE,
		shuffle=True,
		#validation_data=(x_test_gray,x_test_color),
		callbacks=[log,checkpoint])
    return model


def predict():
    # for patch based approach
    patch_size = 9
    stride = 9
    
    input_shape = (patch_size,patch_size,3)
    model = build_model(input_shape)
    if PRETRAINED_MODEL_PATH != NA:
        model.load_weights(PRETRAINED_MODEL_PATH)
        print 'Loaded the model {0}.'.format(PRETRAINED_MODEL_PATH)
    else:
        model.load_weights(MODEL_PATH)
        print 'Loaded the model {0}.'.format(MODEL_PATH)

    num_files = 0
    #dataset_root = '/home/go/work/research/colorization/data/Train_gray'
    #dataset_root = '/home/go/work/research/colorization/data/Validation_gray'
    dataset_root = '/home/go/work/research/colorization/data/Test_gray'
    for root,dirs,files in os.walk(os.path.join(dataset_root)):
        for f in files:
            # ignore files expect png files
            if not f.endswith('png'):
                continue
            gray_image = cv2.imread(os.path.join(root,f))
            shp = gray_image.shape
            dtype = gray_image.dtype
            # padding to align original size sliding with PATCH_SIZExPATCH_SIZE
            # height
            pad_height = 0 if shp[0] % patch_size == 0 else ((shp[0] / patch_size + 1) * patch_size) - shp[0]
            # width
            pad_width = 0 if shp[1] % patch_size == 0 else ((shp[1] / patch_size + 1) * patch_size) - shp[1]

            gray_image = np.pad(gray_image, ((0,pad_height),(0,pad_width),(0,0)), 'constant')
            new_shp = gray_image.shape

            gray_patch_all = []
            for y in xrange(0, new_shp[0], stride):
                for x in xrange(0, new_shp[1], stride):
                    gray_patch = gray_image[y:y+patch_size,x:x+patch_size,:]
                    gray_patch = gray_patch.astype('float32') / 255.
                    gray_patch_all.append(gray_patch)
            # prediction
            print 'Predicting..'
            color_patch_all = model.predict([gray_patch_all])
            print 'Predicted.'
            # reconstruction
            ind = 0
            color_data = np.zeros(new_shp, dtype=dtype)
            for y in xrange(0, new_shp[0], stride):
                for x in xrange(0, new_shp[1], stride):
                    color_data[y:y+patch_size,x:x+patch_size,:] = (color_patch_all[ind] * 255.).astype(dtype)
                    ind += 1
            color_data = color_data[0:shp[0],0:shp[1],:]
            color_data = cv2.cvtColor(color_data, cv2.COLOR_LAB2BGR)
            cv2.imwrite(os.path.join(OUT_PATH,f), color_data)
            num_files += 1
    return num_files

'''
def predict():
    # for patch based approach
    patch_size = 9
    stride = 9
    
    input_shape = (patch_size,patch_size,3)
    model_L = build_model(input_shape)
    model_a = build_model(input_shape)
    model_b = build_model(input_shape)
    model_L.load_weights('runs/23/weights-03.h5')
    model_a.load_weights('runs/24/weights-03.h5')
    model_b.load_weights('runs/25/weights-03.h5')

    num_files = 0
    dataset_root = '/home/go/work/research/colorization/data/Validation_gray'
    for root,dirs,files in os.walk(os.path.join(dataset_root)):
        for f in files:
            # ignore files expect png files
            if not f.endswith('png'):
                continue
            gray_image = cv2.imread(os.path.join(root,f))
            shp = gray_image.shape
            dtype = gray_image.dtype
            # padding to align original size sliding with PATCH_SIZExPATCH_SIZE
            # height
            pad_height = 0 if shp[0] % patch_size == 0 else ((shp[0] / patch_size + 1) * patch_size) - shp[0]
            # width
            pad_width = 0 if shp[1] % patch_size == 0 else ((shp[1] / patch_size + 1) * patch_size) - shp[1]

            gray_image = np.pad(gray_image, ((0,pad_height),(0,pad_width),(0,0)), 'constant')
            new_shp = gray_image.shape

            gray_patch_all = []
            for y in xrange(0, new_shp[0], stride):
                for x in xrange(0, new_shp[1], stride):
                    gray_patch = gray_image[y:y+patch_size,x:x+patch_size,:]
                    gray_patch = gray_patch.astype('float32') / 255.
                    gray_patch_all.append(gray_patch)
            # prediction
            print 'Predicting..'
            color_patch_all_L = model_L.predict([gray_patch_all])
            color_patch_all_a = model_a.predict([gray_patch_all])
            color_patch_all_b = model_b.predict([gray_patch_all])
            print 'Predicted.'
            # reconstruction
            ind = 0
            color_data_L = np.zeros(new_shp, dtype=dtype)
            color_data_a = np.zeros(new_shp, dtype=dtype)
            color_data_b = np.zeros(new_shp, dtype=dtype)
            color_data = np.zeros(shp, dtype=dtype)
            for y in xrange(0, new_shp[0], stride):
                for x in xrange(0, new_shp[1], stride):
                    color_data_L[y:y+patch_size,x:x+patch_size,:] = (color_patch_all_L[ind] * 255.).astype(dtype)
                    color_data_a[y:y+patch_size,x:x+patch_size,:] = (color_patch_all_a[ind] * 255.).astype(dtype)
                    color_data_b[y:y+patch_size,x:x+patch_size,:] = (color_patch_all_b[ind] * 255.).astype(dtype)
                    ind += 1
            color_data_L = color_data_L[0:shp[0],0:shp[1],:]
            color_data_a = color_data_a[0:shp[0],0:shp[1],:]
            color_data_b = color_data_b[0:shp[0],0:shp[1],:]
            color_data[:,:,0] = color_data_L[:,:,0]
            color_data[:,:,1] = color_data_a[:,:,0]
            color_data[:,:,2] = color_data_b[:,:,0]
            color_data = cv2.cvtColor(color_data, cv2.COLOR_LAB2BGR)
            cv2.imwrite(os.path.join(OUT_PATH,f), color_data)
            num_files += 1
    return num_files
'''

def main():
    global RUN,ROUTINGS,N_CLASS,OPTIMIZER,LOSS,EPOCHS,BATCH_SIZE,DATASET,DATA_PATH,PRETRAINED_MODEL_PATH,RUN_PATH,MODEL_PATH,MODEL_SUMMARY_PATH,OUT_PATH,LOG_PATH

    parser = argparse.ArgumentParser(description="Capsule Network based Image Denoiser.")
    parser.add_argument('--train','-tr', action='store_true')
    parser.add_argument('--predict','-pr', action='store_true')
    parser.add_argument('--run','-r', default=1, type=int)
    parser.add_argument('--routings','-ro', default=3, type=int)
    parser.add_argument('--complexity','-c', default=10, type=int)
    parser.add_argument('--optimizer','-o', default='adam', type=str)
    parser.add_argument('--loss','-l', default='mse', type=str)
    parser.add_argument('--epochs','-e', default=10, type=int)
    parser.add_argument('--batch_size','-bs', default=128,type=int)
    parser.add_argument('--dataset','-ds', default='ntire',type=str)
    parser.add_argument('--datapath','-dp', default='../data/train_128_128.npz',type=str)
    parser.add_argument('--pretrained_model','-lm',default=NA,type=str)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    RUN = args.run
    ROUTINGS = args.routings
    N_CLASS = args.complexity
    OPTIMIZER = args.optimizer
    LOSS = args.loss
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    DATASET = args.dataset
    DATA_PATH = args.datapath
    PRETRAINED_MODEL_PATH = args.pretrained_model

    RUN_PATH = os.path.join('runs', str(RUN))
    MODEL_PATH = os.path.join(RUN_PATH, 'weights-'+str(EPOCHS)+'.h5')
    MODEL_SUMMARY_PATH = os.path.join(RUN_PATH, 'model_summary.txt')
    OUT_PATH = os.path.join(RUN_PATH,'out')
    LOG_PATH = os.path.join(RUN_PATH, 'log.csv')

    print args

    if not os.path.exists(RUN_PATH):
        os.makedirs(RUN_PATH)
    if not os.path.exists(OUT_PATH):
        os.makedirs(OUT_PATH)

    f = open(os.path.join('runs', 'table.txt'),'a')
    f.write(str(RUN)+'\n')
    f.write(parser.prog+'\n')
    f.write(str(args)+'\n\n\n')
    f.close()

    if args.train:
        data = load_ntire()
        model = train(data)
        if args.save:
            model.save_weights(MODEL_PATH)
    elif args.predict:
        tic = timeit.default_timer()
        num_files = predict()
        toc = timeit.default_timer()
        print 'Runtime per image [s] : {0}'.format((toc-tic) / (num_files*1.0))
        
if __name__ == '__main__':
    main()
