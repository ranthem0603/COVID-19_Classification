#%%
import numpy as np
import os         
import matplotlib.pyplot as plt             
import cv2                                 
from sklearn.preprocessing import LabelEncoder , OneHotEncoder
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from matplotlib.pyplot import imshow
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, auc
#%%
covid_types =  os.listdir(r'datapath')
covid= []
for item in covid_types:
 # Get all the file names
 all_covid = os.listdir(r'datapath' + '/' +item)
 # Add them to the list
 for covids in all_covid:
    covid.append((item, str(r'datapath' + '/' +item) + '/' + covids))
#%%
covid_df = pd.DataFrame(data=covid, columns=['covid type', 'image'])
covid_count = covid_df['covid type'].value_counts()
#%%
path = r'datapath'
covid_types =  os.listdir(r'datapath')
#%%
im_size = 224
images = []
labels = []
for i in covid_types:
    data_path = path + str(i)  # entered in 1st folder and then 2nd folder and then 3rd folder
    filenames = [i for i in os.listdir(data_path) ]
   # print(filenames)  # will get the names of all images
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)  # reading that image as array
        #print(img)  # will get the image as an array
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)
labels   
#%%
# Transform the image array to a numpy type
images = np.array(images)
images.shape   
#%%
images = images.astype('float32') / 255.0
#%%
y=covid_df['covid type'].values
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)
#%%
images, y = shuffle(images, y, random_state=1)
#%%
print(images)
#%%
train_x, test_x, train_y, test_y = train_test_split(images, y, test_size=0.1, random_state=415)
#%%
def identity_block(X, f, filters, stage, block):
    """"
    ----------
    Receives
    X : tensor, input tensor of shape
    f : integer, specifying the shape of the middle CONV's window for the main path
    filters : list, python list of integers, defining the number of filters in the CONV layers of the main path
    stage : integer, used to name the layers, depending on their position in the network
    block : str, used to name the layers, depending on their position in the network
    -------
    Returns
    X : tensor, output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. we'll need this later to add back to the main path. 
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
               name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', 
               name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', 
               name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
#%%
def convolutional_block(X, f, filters, stage, block, s=2):
    """
    ----------
    Receives
    X : tensor, input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f : integer, specifying the shape of the middle CONV's window for the main path
    filters : list, python list of integers, defining the number of filters in the CONV layers of the main path
    stage : integer, used to name the layers, depending on their position in the network
    block : str, used to name the layers, depending on their position in the network
    s : integer, optional: Integer, specifying the stride to be used. The default is 2.
    -------
    Returns
    X : tensor, output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X
#%%
def ResNet50(input_shape, outputClasses):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    ----------
    Receives
    input_shape : tuple, optional:shape of the input image. 
    outputClasses : integer, optional: number of classes. 
    -------
    Returns
    model : object, a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3 
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='c')
    X = identity_block(X, 3, [128, 128, 512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL 
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(outputClasses, activation='softmax', name='fc' + str(outputClasses), 
              kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model

#%%
model = ResNet50(input_shape = (224, 224, 3), outputClasses = 13)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
train_y = train_y.astype('float32')
history = model.fit(train_x, train_y, batch_size=32, epochs=49, validation_data=(test_x, test_y))
preds = model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
y_pred = model.predict(test_x).argmax(axis=1)

target_names = ['B.1.1.7-mix','B.1.351-mix','B.1.429-mix'
                ,'B.1.525-mix','B.1.526-mix','B.1.617.1-mix','B.1.617.2-mix'
                ,'C.37-mix','P.1-mix','P.2-black','B.1.640.1-mix','B.1.640.2-mix','B.1.351.1-mix']
print(classification_report(test_y, y_pred, target_names=target_names))
#%%
cm = confusion_matrix(test_y, y_pred)
print(cm)
#%%
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()
#%%
y_pred_proba = model.predict_on_batch(test_x)
y_pred_proba
#%%
roc_auc_score(test_y,y_pred_proba,multi_class="ovr")
#%%
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.show()
#%%
plt.plot(history.history['accuracy'])
plt.grid()
plt.title('Model Train Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train'])
plt.show()
#%%
plt.plot(history.history['val_accuracy'],color='darkorange')
plt.grid()
plt.title('Model validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Validation'])
plt.show()
#%%
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.grid()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.show()
#%%
plt.plot(history.history['loss'])
plt.grid()
plt.title('Model Train Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Train'])
plt.show()
#%%
plt.plot(history.history['val_loss'],color='darkorange')
plt.grid()
plt.title('Model Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['Validation'])
plt.show()