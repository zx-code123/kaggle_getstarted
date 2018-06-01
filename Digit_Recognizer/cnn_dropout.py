'''
Created on 2018-5-31
Author: zx-code123
Github: https://github.com/zx-code123
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')
# Load the data
def load_data():
    train_data = pd.read_csv("digit_data/train.csv")
    test_data = pd.read_csv("digit_data/test.csv")
    X_train = (train_data.iloc[:,1:].values).astype('float32')
    y_train = (train_data.iloc[:,0].values).astype('int32')
    X_test = test_data.astype('float32')

    print("The MNIST dataset has a training set of {} examples.".format(X_train.shape))
    print("The MNIST database has a test set of {} examples.".format(X_test.shape))
    # visualize the data (see detail of each pixel)
    X_train = X_train.reshape(X_train.shape[0],28,28)
    X_test = X_test.values.reshape(-1,28,28,1)
    visualize(X_train,y_train)
    # 增加一个维度
    X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
    print(X_test.shape,X_train.shape)
    # rescale the images (/255)
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    return X_train,X_test,y_train

# 数据可视化
def visualize(data,data_label):
    fig = plt.figure(figsize=(6,4))
    for i in range(6):
        ax = plt.subplot(2, 3, i+1)
        ax.axis('off')
        ax.set_title('label: {}'.format(data_label[i]))
        plt.imshow(data[i],cmap = plt.get_cmap('gray'))
    plt.show()

# 处理数据，对数据分割训练集，验证集，将标签转成one-hot 
def data_preprocess(X_train,y_train):
    from keras.utils import np_utils
    from sklearn.model_selection import train_test_split
    # Set the random seed
    random_seed = 2
    # Split the train and the validation set for the fitting
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state=random_seed)
    # print first ten (integer-valued) training labels
    print('Integer-valued labels:')
    print(y_train[:10])

    # one-hot encode the labels
    y_train = np_utils.to_categorical(y_train, 10)
    y_val = np_utils.to_categorical(y_val, 10)

    # print first ten (one-hot) training labels
    print('One-hot labels:')
    print(y_train[:10])
    return X_train, X_val, y_train, y_val
# 定义模型
def cnnModel(X_train):
    # Set the CNN model 
    # my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

    model = Sequential()

    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                    activation ='relu', input_shape = (28,28,1)))
    model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))
    model.add(Dropout(0.25))


    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                    activation ='relu'))
    model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
    model.add(Dropout(0.25))


    model.add(Flatten())
    model.add(Dense(256, activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation = "softmax"))
    return model

# Look at confusion matrix 

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
# 没有进行data augmentation
def main():
    X_train,X_test,y_train = load_data()
    X_train, X_val, Y_train, Y_val = data_preprocess(X_train,y_train)
    model = cnnModel(X_train)
    # Define the optimizer
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, 
                metrics=['accuracy'])
    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
    epochs = 30 # Turn epochs to 30 to get 0.9967 accuracy
    batch_size = 86
    #don't apply a vertical_flip nor horizontal_flip since 
    #it could lead to misclassify symetrical numbers such as 6 and 9.
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


    datagen.fit(X_train) 

    # Fit the model
    history = model.fit_generator(datagen.flow(X_train,Y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (X_val,Y_val),
                              verbose = 1, steps_per_epoch=X_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])

    # Plot the loss and accuracy curves for training and validation 
    fig, ax = plt.subplots(2,1)
    ax[0].plot(history.history['loss'], color='b', label="Training loss")
    ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
    legend = ax[0].legend(loc='best', shadow=True)

    ax[1].plot(history.history['acc'], color='b', label="Training accuracy")
    ax[1].plot(history.history['val_acc'], color='r',label="Validation accuracy")
    legend = ax[1].legend(loc='best', shadow=True)
    plt.show()

    # Predict the values from the validation dataset
    Y_pred = model.predict(X_val)
    # Convert predictions classes to one hot vectors 
    Y_pred_classes = np.argmax(Y_pred,axis = 1) 
    # Convert validation observations to one hot vectors
    Y_true = np.argmax(Y_val,axis = 1) 
    # compute the confusion matrix
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    # plot the confusion matrix
    plot_confusion_matrix(confusion_mtx, classes = range(10)) 

    results = model.predict(X_test)
    results = np.argmax(results,axis = 1)
    results = pd.Series(results,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv("keras_cnn.csv",index=False)

if __name__ =='__main__':
    main()