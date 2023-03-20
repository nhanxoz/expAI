import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.models import load_model
import sys
import h5py
from expAI.models import *
from tensorflow.keras import callbacks
import json

def train(para_id,dataset_path,json_config):
    json_config = json.loads(json_config)
    epoch = json_config['epoch']
    batchS = json_config['batch_size']
    class CustomCallback(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            _para = Paramsconfigs.objects.get(pk=para_id)
            if _para.trainningstatus == 0:
                _new_result = Trainningresults()
                _new_result.configid = _para
                _new_result.accuracy = float(logs['accuracy'])
                _new_result.lossvalue = float(logs['loss'])
                _new_result.trainresultindex = epoch+1
                _new_result.is_last = True
                _new_result.save()
                self.model.stop_training = True

            else:
                _new_result = Trainningresults()
                _new_result.configid = _para
                _new_result.accuracy = float(logs['accuracy'])
                _new_result.lossvalue = float(logs['loss'])
                _new_result.trainresultindex = epoch+1
                _new_result.is_last = False  
                _new_result.save()

    
    in_dir = './datasets/' + dataset_path + "/data"
    # Frame size  
    img_size = 224
    img_size_touple = (img_size, img_size)
    # Number of channels (RGB)
    num_channels = 3
    # Flat frame size
    img_size_flat = img_size * img_size * num_channels
    # Number of classes for classification (Violence-No Violence)
    num_classes = 2
    # Number of files to train
    _num_files_train = 1
    # Number of frames per video
    _images_per_file = 20
    # Number of frames per training set
    _num_images_train = _num_files_train * _images_per_file
    # Video extension
    video_exts = ".avi"
    # First get the names and labels of the whole videos
    names, labels = label_video_names(in_dir)
    image_model = MobileNet( weights='imagenet',include_top=True,input_shape=(224,224,3))
    image_model.summary()
    transfer_layer = image_model.get_layer('reshape_2')

    image_model_transfer = Model(inputs=image_model.input,
                                outputs=transfer_layer.output)
    transfer_values_size = K.int_shape(transfer_layer.output)[1]
    print("The input of the VGG16 net have dimensions:",K.int_shape(image_model.input)[1:3])

    print("The output of the selecter layer of VGG16 net have dimensions: ", transfer_values_size)

    training_set = int(len(names))

    make_files(training_set,names,in_dir,labels, _images_per_file, img_size_touple,transfer_values_size,image_model_transfer,img_size)
    data, target = process_alldata_training(in_dir)
    chunk_size = 1000
    n_chunks = 20
    rnn_size = 512
    model = Sequential()
    model.add(LSTM(rnn_size, input_shape=(n_chunks, chunk_size)))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])
    history = model.fit(np.array(data[0:720]), np.array(target[0:720]), epochs=epoch,
                        validation_data=(np.array(data[720:]), np.array(target[720:])), 
                        batch_size=batchS, verbose=2, callbacks=[CustomCallback()],)
    
    
    _para = Paramsconfigs.objects.get(pk=para_id)
    save_model_path = './weights/' + str(_para.pk)
    if not os.path.exists(save_model_path):
        # Create a new directory because it does not exist
        os.makedirs(save_model_path)
    _para.trainningstatus = 0
    _para.configaftertrainmodelpath = save_model_path + '/model.h5'
    _para.save()
    model.save(save_model_path + '/model.h5')
    print('Model Saved!')
    
def test(ressult_id,dataset_path):
    _ressult = Results.objects.get(pk = ressult_id)
    _para = Paramsconfigs.objects.get(pk=_ressult.resultconfigid.pk)

    # load model
    model=load_model(_para.configaftertrainmodelpath)
    model.summary()
    
    in_dir = './datasets/' + dataset_path + "/data"
    # Frame size  
    img_size = 224
    img_size_touple = (img_size, img_size)
    # Number of channels (RGB)
    num_channels = 3
    # Flat frame size
    img_size_flat = img_size * img_size * num_channels
    # Number of classes for classification (Violence-No Violence)
    num_classes = 2
    # Number of files to train
    _num_files_train = 1
    # Number of frames per video
    _images_per_file = 20
    # Number of frames per training set
    _num_images_train = _num_files_train * _images_per_file
    # Video extension
    video_exts = ".avi"
    # First get the names and labels of the whole videos
    names, labels = label_video_names(in_dir)

    image_model = MobileNet( weights='imagenet',include_top=True,input_shape=(224,224,3))
    image_model.summary()
    transfer_layer = image_model.get_layer('reshape_2')

    image_model_transfer = Model(inputs=image_model.input,
                                outputs=transfer_layer.output)
    
    transfer_values_size = K.int_shape(transfer_layer.output)[1]


    print("The input of the VGG16 net have dimensions:",K.int_shape(image_model.input)[1:3])

    print("The output of the selecter layer of VGG16 net have dimensions: ", transfer_values_size)

    test_set = int(len(names))
    make_files_test(test_set,names,in_dir,labels, _images_per_file, img_size_touple,transfer_values_size,image_model_transfer,img_size)
    data, target = process_alldata_training(in_dir)
    result = model.evaluate(np.array(data), np.array(target))
    _ressult.resultaccuracy = float(result[1])
    _ressult.resultdetail ="loss: " + str(result[0]) + "acc: " + str(result[1])
    _ressult.save()


def predict(pre_id,trained_model):
    _pre = Predict.objects.get(pk=pre_id)
    result_path = str(_pre.inputpath)[:9] + 'predict_result' + str(_pre.inputpath)[21:]
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    _pre.outputpath = result_path
    _pre.save()
    # load model
    model=load_model(trained_model)
    model.summary()
    in_dir = str(_pre.inputpath)
    # Frame size  
    img_size = 224
    img_size_touple = (img_size, img_size)
    # Number of channels (RGB)
    num_channels = 3
    # Flat frame size
    img_size_flat = img_size * img_size * num_channels
    # Number of classes for classification (Violence-No Violence)
    num_classes = 2
    # Number of files to train
    _num_files_train = 1
    # Number of frames per video
    _images_per_file = 20
    # Number of frames per training set
    _num_images_train = _num_files_train * _images_per_file
    # Video extension
    video_exts = ".avi"

    image_model = MobileNet( weights='imagenet',include_top=True,input_shape=(224,224,3))
    image_model.summary()
    transfer_layer = image_model.get_layer('reshape_2')

    image_model_transfer = Model(inputs=image_model.input,
                                outputs=transfer_layer.output)
    
    transfer_values_size = K.int_shape(transfer_layer.output)[1]


    print("The input of the VGG16 net have dimensions:",K.int_shape(image_model.input)[1:3])

    print("The output of the selecter layer of VGG16 net have dimensions: ", transfer_values_size)

    for video_name in os.listdir(in_dir):
        video_path = os.path.join(in_dir,video_name)
        images = []
        vidcap = cv2.VideoCapture(video_path)
        frame_width = int(vidcap.get(3))
        frame_height = int(vidcap.get(4))    
        size = (frame_width, frame_height)
        out_name = os.path.join(str(_pre.outputpath),os.path.splitext(video_name)[0]+'.avi')
        print(out_name)
        output = cv2.VideoWriter(out_name, 
                            cv2.VideoWriter_fourcc(*'MJPG'),
                            10, size)
        success,image = vidcap.read()
        count = 0
        while success:
                    
            RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
            res = cv2.resize(RGB_img, dsize=(img_size, img_size),
                                    interpolation=cv2.INTER_CUBIC)
        
            images.append(res)
        
            success,image = vidcap.read()
            if count > 20 :
                resul = np.array(images[(count-20):count])
                resul = (resul / 255.).astype(np.float16)
                shape = (_images_per_file,) + img_size_touple + (3,)
                    # Pre-allocate input-batch-array for images.
                image_batch = np.zeros(shape=shape, dtype=np.float16)
                
                image_batch = resul
                # Pre-allocate output-array for transfer-values.
                # Note that we use 16-bit floating-points to save memory.
                shape = (_images_per_file, transfer_values_size)
                transfer_values = np.zeros(shape=shape, dtype=np.float16)

                transfer_values = \
                        image_model_transfer.predict(image_batch)
                result = model.predict(np.array([transfer_values]))
                if np.argmax(result[0]) == 0:
                    image = cv2.putText(
                        img = image,
                        text = "Fightting!",
                        org = (int(frame_width/5), int(frame_height/5)),
                        fontFace = cv2.FONT_HERSHEY_DUPLEX,
                        fontScale = 1.0,
                        color = (0,0,255),
                        thickness = 2
                        )
                    
            if count > 1000:
                break
            output.write(image)

            count += 1


def print_progress(count, max_count):
    # Percentage completion.
    pct_complete = count / max_count

    # Status-message. Note the \r which means the line should
    # overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()

def get_frames(current_dir, file_name, _images_per_file,img_size):
    
    in_file = os.path.join(current_dir, file_name)
    
    images = []
    
    vidcap = cv2.VideoCapture(in_file)
    
    success,image = vidcap.read()
        
    count = 0

    while count<_images_per_file:
                
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        res = cv2.resize(RGB_img, dsize=(img_size, img_size),
                                 interpolation=cv2.INTER_CUBIC)
    
        images.append(res)
    
        success,image = vidcap.read()
    
        count += 1
        
    resul = np.array(images)
    
    resul = (resul / 255.).astype(np.float16)
        
    return resul

def label_video_names(in_dir):
    
    # list containing video names
    names = []
    # list containin video labels [1, 0] if it has violence and [0, 1] if not
    labels = []
    for current_dir, dir_names,file_names in os.walk(in_dir):
        
        for file_name in file_names:
            
            if file_name[0:2] == 'fi':
                labels.append([1,0])
                names.append(file_name)
            elif file_name[0:2] == 'no':
                labels.append([0,1])
                names.append(file_name)
                     
    c = list(zip(names,labels))
    # Suffle the data (names and labels)
    shuffle(c)
    
    names, labels = zip(*c)
            
    return names, labels

def get_transfer_values(current_dir, file_name,_images_per_file,img_size_touple,transfer_values_size, image_model_transfer):
    
    # Pre-allocate input-batch-array for images.
    shape = (_images_per_file,) + img_size_touple + (3,)
    
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    
    image_batch = get_frames(current_dir, file_name)
      
    # Pre-allocate output-array for transfer-values.
    # Note that we use 16-bit floating-points to save memory.
    shape = (_images_per_file, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    transfer_values = \
            image_model_transfer.predict(image_batch)
            
    return transfer_values

def proces_transfer(vid_names, in_dir, labels, _images_per_file, img_size_touple,transfer_values_size,image_model_transfer, img_size):
    
    count = 0
    
    tam = len(vid_names)
    
    # Pre-allocate input-batch-array for images.
    shape = (_images_per_file,) + img_size_touple + (3,)
    
    while count<tam:
        
        video_name = vid_names[count]
        
        image_batch = np.zeros(shape=shape, dtype=np.float16)
    
        image_batch = get_frames(in_dir, video_name,_images_per_file,img_size)
        
         # Note that we use 16-bit floating-points to save memory.
        shape = (_images_per_file, transfer_values_size)
        transfer_values = np.zeros(shape=shape, dtype=np.float16)
        
        transfer_values = \
            image_model_transfer.predict(image_batch)
         
        labels1 = labels[count]
        
        aux = np.ones([20,2])
        
        labelss = labels1*aux
        
        yield transfer_values, labelss
        
        count+=1

def make_files(n_files,names_training,in_dir,labels_training, _images_per_file, img_size_touple,transfer_values_size,image_model_transfer,img_size):
    
    gen = proces_transfer(names_training, in_dir, labels_training, _images_per_file, img_size_touple,transfer_values_size,image_model_transfer,img_size)

    numer = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    
    with h5py.File(in_dir + '/prueba.h5', 'w') as f:
    
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
    
    
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
    
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)
    
         # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            
            if numer == n_files:
            
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            
            print_progress(numer, n_files)
        
            numer += 1

def make_files_test(n_files,names_test,in_dir,labels_test, _images_per_file, img_size_touple,transfer_values_size,image_model_transfer, img_size):
    
    gen = proces_transfer(names_test, in_dir, labels_test, _images_per_file, img_size_touple,transfer_values_size,image_model_transfer,img_size)

    numer = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    
    with h5py.File(in_dir + '/pruebavalidation.h5', 'w') as f:
    
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
    
    
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
    
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)
    
         # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            
            if numer == n_files:
            
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            
            print_progress(numer, n_files)
        
            numer += 1

def process_alldata_training(indir):
    
    joint_transfer=[]
    frames_num=20
    count = 0
    
    with h5py.File(indir + '/prueba.h5', 'r') as f:
            
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch)/frames_num)):
        inc = count+frames_num
        joint_transfer.append([X_batch[count:inc],y_batch[count]])
        count =inc
        
    data =[]
    target=[]
    
    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))
        
    return data, target

def process_alldata_test(indir):
    
    joint_transfer=[]
    frames_num=20
    count = 0
    
    with h5py.File(indir + '/pruebavalidation.h5', 'r') as f:
            
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch)/frames_num)):
        inc = count+frames_num
        joint_transfer.append([X_batch[count:inc],y_batch[count]])
        count =inc
        
    data =[]
    target=[]
    
    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))
        
    return data, target

if __name__ == "__main__":
    predict()