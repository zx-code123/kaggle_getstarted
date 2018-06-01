'''
Created on 2018-5-31
Author: zx-code123
Github: https://github.com/zx-code123
'''
'''
Steps:
import needed libraries
Load the data
visualize the data (see detail of each pixel)
rescale the images (/255)
One-Hot Scheme
Model Architecture
Compile the model
calculate accuracy before the training
train the model
calculate the accuracy after training 10.1. implement data augmentation and compare results
output the submission file
'''
#%% import needed libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
def mlpModel(X_train):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten

    # define the model
    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))

    # summarize the model
    model.summary()
    print("input shape ",model.input_shape)
    print("output shape ",model.output_shape)
    return model
# 没有进行data augmentation
def main():
    X_train,X_test,y_train = load_data()
    X_train, X_val, y_train, y_val = data_preprocess(X_train,y_train)
    model = mlpModel(X_train)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                metrics=['accuracy'])

    from keras.callbacks import ModelCheckpoint
    from keras.preprocessing.image import ImageDataGenerator
    #don't apply a vertical_flip nor horizontal_flip since 
    #it could lead to misclassify symetrical numbers such as 6 and 9.
    dataAug = ImageDataGenerator(
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
    dataAug.fit(X_train)   

    # train the model
    checkpointer = ModelCheckpoint(filepath='mnist_model.hdf5', 
                                verbose=1, save_best_only=True)
    hist = model.fit_generator(dataAug.flow(X_train, y_train, batch_size=128), epochs=20,
          validation_data = (X_val,y_val), callbacks=[checkpointer],
          verbose=1)

    # evaluate test accuracy
    score = model.evaluate(X_val, y_val, verbose=1)
    accuracy = 100*score[1]

    # print test accuracy
    print('validation accuracy: %.4f%%' % accuracy)
    results = model.predict(X_test)
    results = np.argmax(results,axis = 1)
    results = pd.Series(results,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv("keras_nn.csv",index=False)
# 没有进行data augmentation
def main_1():
    X_train,X_test,y_train = load_data()
    X_train, X_val, y_train, y_val = data_preprocess(X_train,y_train)
    model = mlpModel(X_train)

    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', 
                metrics=['accuracy'])

    from keras.callbacks import ModelCheckpoint   

    # train the model
    checkpointer = ModelCheckpoint(filepath='mnist_model.hdf5', 
                                verbose=1, save_best_only=True)
    hist = model.fit(X_train, y_train, batch_size=128, epochs=20,
            validation_split=0.2, callbacks=[checkpointer],
            verbose=1, shuffle=True)

    # evaluate test accuracy
    score = model.evaluate(X_val, y_val, verbose=1)
    accuracy = 100*score[1]

    # print test accuracy
    print('validation accuracy: %.4f%%' % accuracy)

if __name__ =='__main__':
    main()

