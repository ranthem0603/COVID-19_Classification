#%%
import numpy as np
import os         
import matplotlib.pyplot as plt             
import cv2                                 
from sklearn.preprocessing import LabelEncoder 
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import layers
from keras.layers import Input, Add, Dense, Activation, BatchNormalization, Conv2D
from keras.models import Model, load_model
from keras.applications.imagenet_utils import preprocess_input
from keras.utils import plot_model
from matplotlib.pyplot import imshow
from keras.initializers import glorot_uniform
from tensorflow.keras.layers import Input, Conv2D, SeparableConv2D, \
     Add, Dense, BatchNormalization, ReLU, MaxPool2D, GlobalAvgPool2D
from tensorflow.keras import Model 
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
#%%
covid_types =  os.listdir(r'datapath')
covid= []
for item in covid_types:
 # Get all the file names
 all_covid = os.listdir(r'datapath' + '/' +item)
 # Add them to the list
 for covids in all_covid:
    covid.append((item, str(r'datapath' + '/' +item) + '/' + covids))

covid_df = pd.DataFrame(data=covid, columns=['covid type', 'image'])

covid_count = covid_df['covid type'].value_counts()

path = r'datapath'
covid_types =  os.listdir(r'datapath')

im_size = 224
images = []
labels = []
#%%
for i in covid_types:
    data_path = path + str(i)  # entered in 1st folder and then 2nd folder and then 3rd folder
    filenames = [i for i in os.listdir(data_path) ]
    # will get the names of all images
    for f in filenames:
        img = cv2.imread(data_path + '/' + f)  # reading that image as array
        img = cv2.resize(img, (im_size, im_size))
        images.append(img)
        labels.append(i)
labels   
#%%
# Transform the image array to a numpy type
images = np.array(images)
images.shape   
images = images.astype('float32') / 255.0

y=covid_df['covid type'].values
y_labelencoder = LabelEncoder ()
y = y_labelencoder.fit_transform (y)

images, y = shuffle(images, y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(images, y, test_size=0.1, random_state=415)
#%%
def conv_bn(x, filters, kernel_size, strides=1):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides,
               padding='same',
               use_bias=False)(x)
    x = BatchNormalization()(x)
    return x

#%%
def sep_bn(x, filters, kernel_size, strides=1):
    x = SeparableConv2D(filters=filters,
                        kernel_size=kernel_size,
                        strides=strides,
                        padding='same',
                        use_bias=False)(x)
    x = BatchNormalization()(x)
    return x
#%%
def entry_flow(x):
    x = conv_bn(x, filters=32, kernel_size=3, strides=2)
    x = ReLU()(x)
    x = conv_bn(x, filters=64, kernel_size=3)
    tensor = ReLU()(x)

    x = sep_bn(tensor, filters=128, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=128, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=128, kernel_size=1, strides=2)

    x = Add()([tensor, x])
    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=256, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=256, kernel_size=1, strides=2)

    x = Add()([tensor, x])
    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = MaxPool2D(pool_size=3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=728, kernel_size=1, strides=2)
    x = Add()([tensor, x])

    return x
#%%
def middle_flow(tensor):
    for _ in range(8):
        x = ReLU()(tensor)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)
        x = ReLU()(x)
        x = sep_bn(x, filters=728, kernel_size=3)

        tensor = Add()([tensor, x])

    return tensor
#%%
def exit_flow(tensor):
    x = ReLU()(tensor)
    x = sep_bn(x, filters=728, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=1024, kernel_size=3)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    tensor = conv_bn(tensor, filters=1024, kernel_size=1, strides=2)

    x = Add()([tensor, x])
    x = sep_bn(x, filters=1536, kernel_size=3)
    x = ReLU()(x)
    x = sep_bn(x, filters=2048, kernel_size=3)
    x = ReLU()(x)
    x = GlobalAvgPool2D()(x)
    x = Dense(units=13, activation='softmax')(x)

    return x
#%%
input = Input(shape=[224, 224, 3])

x = entry_flow(input)
x = middle_flow(x)
output = exit_flow(x)

model = Model(input, output)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_y = train_y.astype('float32')

history = model.fit(train_x, train_y, batch_size=32, epochs=49, validation_data=(test_x, test_y))
#%%
preds = model.evaluate(test_x, test_y)
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))
y_pred = model.predict(test_x).argmax(axis=1)

target_names = ['B.1.1.7-mix','B.1.351-mix','B.1.429-mix'
                ,'B.1.525-mix','B.1.526-mix','B.1.617.1-mix','B.1.617.2-mix'
                ,'C.37-mix','P.1-mix','P.2-black','B.1.640.1-mix','B.1.640.2-mix','B.1.351.1-mix']
print(classification_report(test_y, y_pred, target_names=target_names))

cm = confusion_matrix(test_y, y_pred)
print(cm)
#%%
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True)
plt.title('Confusion Matrix')
plt.ylabel('Actal Values')
plt.xlabel('Predicted Values')
plt.show()

y_pred_proba = model.predict_on_batch(test_x)
y_pred_proba

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
