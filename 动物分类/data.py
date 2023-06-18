import cv2
import torch
import numpy as np

test_size = 100
train_size = 1000
pic_size = 128
train_set = np.zeros(3*train_size*2*pic_size*pic_size)
test_set = np.zeros(3*test_size*2*pic_size*pic_size)
test_set = np.reshape(test_set,(test_size*2,3,pic_size,pic_size))
train_set = np.reshape(train_set,(train_size*2,3,pic_size,pic_size))
train_target = np.zeros(train_size*2)
test_target = np.zeros(test_size*2)

sucess_mark = 0
for i in range(train_size):
    path1 = f"data/training_set/cats/cat.{i+1}.jpg"
    path2 = f"data/training_set/dogs/dog.{i+1}.jpg"
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.resize(img1,(pic_size,pic_size))
    img2 = cv2.resize(img2,(pic_size,pic_size))
    train_set[2*i,0,:,:] = img1[:,:,0]
    train_set[2*i,1,:,:] = img1[:,:,1]
    train_set[2*i,2,:,:] = img1[:,:,2]
    sucess_mark +=1
    train_set[2*i+1, 0, :, :] = img2[:, :, 0]
    train_set[2*i+1, 1, :, :] = img2[:, :, 1]
    train_set[2*i+1, 2, :, :] = img2[:, :, 2]
    sucess_mark += 1

for i in range(test_size):
    path1 = f"data/test_set/cats/cat.{i+4001}.jpg"
    path2 = f"data/test_set/dogs/dog.{i+4001}.jpg"
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    img1 = cv2.resize(img1,(pic_size,pic_size))
    img2 = cv2.resize(img2,(pic_size,pic_size))
    test_set[2*i,0,:,:] = img1[:,:,0]
    test_set[2*i,1,:,:] = img1[:,:,1]
    test_set[2*i,2,:,:] = img1[:,:,2]
    sucess_mark +=1
    test_set[2*i+1, 0, :, :] = img2[:, :, 0]
    test_set[2*i+1, 1, :, :] = img2[:, :, 1]
    test_set[2*i+1, 2, :, :] = img2[:, :, 2]
    sucess_mark += 1

if sucess_mark == (test_size + train_size)*2:
    np.save("train_set.npy",train_set)
    np.save("test_set.npy",test_set)