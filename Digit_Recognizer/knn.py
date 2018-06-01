'''
Created on 2018-5-31
Author: zx-code123
Github: https://github.com/zx-code123
'''
import os
import csv
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

dataset_dir = "digit_data"

# 读取csv文件
def opencsv():
    # 使用pandas读取
    data_train = pd.read_csv(os.path.join(dataset_dir,"train.csv"))
    data_test = pd.read_csv(os.path.join(dataset_dir,"test.csv"))

    train_df = data_train.values[0:,1:] #读取训练数据
    test_df = data_test.values[0:,0:] # 读取测试数据
    label_df = data_train.values[0:,0] # 读取训练标签
    return train_df,label_df,test_df

def knnClassify(train_df,label_df):
    knnClf = KNeighborsClassifier(n_neighbors=5)
    knnClf.fit(train_df,label_df)
    return knnClf


# 保存方法一
def saveResult(result,outputName):
    # 创建记录输出结果的文件（w 和 wb 使用的时候有问题）
    with open(outputName,"w") as f:
        # python3里面对 str和bytes类型做了严格的区分，不像python2里面某些函数里可以混用。所以用python3来写wirterow时，打开文件不要用wb模式，只需要使用w模式，然后带上newline=''
        output = csv.writer(f)
        output.writerow(["ImageId","Label"])
        index = 0
        for r in result:
            index += 1
            output.writerow([index,int(r)])
        print("[INFO] save succcessfully")  
        
def main():
    start_time = time.time()
    # 读取数据
    train_df,label_df,test_df = opencsv()
    s_time1 = time.time()
    print("[INFO] load data cost:{}".format(s_time2-start_time))

    # 训练模型
    knnClf = knnClassify(train_df,label_df)
    # 预测结果
    test_label = knnClf.predict(test_df)
    saveResult(test_label,"result.csv")
    # 保存方法二
    # pd.DataFrame({"ImageId": range(1,len(test_label)+1), "Label": test_label}).to_csv('result.csv', index=False, header=True)
    s_time2 = time.time()
    print("[INFO] classify test_data cost:{}".format(s_time2-s_time1))

if __name__ =='__main__':
    main()

