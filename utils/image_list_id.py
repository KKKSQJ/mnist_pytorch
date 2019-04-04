#coding:utf8
import os
import random
from tqdm import tqdm

# 遍历目录，得到所有文件
def traversPath(rootDir, fileList):
    for root, dirs, files in os.walk(rootDir):
        for file in files:
            #fileList.append(os.path.join(root, file))
            fileList.append(root+"/"+file)
        for dir in dirs:
            traversPath(dir, fileList)

# 图像列表shuffle, 分为train和val集合
def shuffle_split(listFile, trainFile, valFile):
    with open(listFile, 'r') as f:
        records = f.readlines()
    random.shuffle(records)
    num = len(records)
    trainNum = int(num*0.8)
    with open(trainFile, 'w') as f1:
        f1.writelines(records[0:trainNum])
    with open(valFile, 'w') as f2:
        f2.writelines(records[trainNum:])

# 获取图像列表和标签TXT
def ImageLabelTXT(fileList, imagelabeltxt):
    with open(imagelabeltxt, 'w') as f:
        for line in tqdm(fileList):
            imagepath = line.strip()
            label = imagepath.split('/')[-2]
            f.write("{} {}\n".format(imagepath, label))
            #f.write("{}\n".format(imagepath))



