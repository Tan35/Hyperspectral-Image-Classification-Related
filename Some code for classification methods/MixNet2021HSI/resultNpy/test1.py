import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.io as sio
import matplotlib
import cv2
if __name__ == '__main__':
    outputs = np.load('saveoutput1.npy')

    outputs = outputs.astype(int)
    plt.imshow(outputs.astype(int))

    plt.show()
    savenum = np.zeros((outputs.max() + 1,2))
    outputs1 = np.zeros((outputs.shape[0],outputs.shape[1],3))
    for i in range(outputs.shape[0]):
        for j in range(outputs.shape[1]):
            savenum[outputs[i,j],0] = savenum[outputs[i,j],0] + 1
            if (outputs[i,j]  == 1):
                outputs1[i,j, 0] = 255
                outputs1[i, j, 1] = 0
                outputs1[i, j, 2] = 0
            elif (outputs[i,j]  == 2):
                outputs1[i,j, 0] = 0
                outputs1[i, j, 1] = 255
                outputs1[i, j, 2] = 0
            elif (outputs[i,j]  == 3):
                outputs1[i,j, 0] = 0
                outputs1[i, j, 1] = 0
                outputs1[i, j, 2] = 255
            elif (outputs[i, j] == 4):
                outputs1[i, j, 0] = 255
                outputs1[i, j, 1] = 255
                outputs1[i, j, 2] = 0
            elif (outputs[i, j] == 5):
                outputs1[i, j, 0] = 255
                outputs1[i, j, 1] = 0
                outputs1[i, j, 2] = 255
            elif (outputs[i, j] == 6):
                outputs1[i, j, 0] = 0
                outputs1[i, j, 1] = 255
                outputs1[i, j, 2] = 255
            elif (outputs[i, j] == 7):
                outputs1[i, j, 0] = 199
                outputs1[i, j, 1] = 99
                outputs1[i, j, 2] = 0
            elif (outputs[i, j] == 8):
                outputs1[i, j, 0] = 0
                outputs1[i, j, 1] = 201
                outputs1[i, j, 2] = 99
            elif (outputs[i, j] == 9):
                outputs1[i, j, 0] = 98
                outputs1[i, j, 1] = 0
                outputs1[i, j, 2] = 198
            elif (outputs[i, j] == 10):
                outputs1[i, j, 0] = 201
                outputs1[i, j, 1] = 0
                outputs1[i, j, 2] = 98
            elif (outputs[i, j] == 11):
                outputs1[i, j, 0] = 98
                outputs1[i, j, 1] = 202
                outputs1[i, j, 2] = 0
            elif (outputs[i, j] == 12):
                outputs1[i, j, 0] = 0
                outputs1[i, j, 1] = 100
                outputs1[i, j, 2] = 201
            elif (outputs[i, j] == 13):
                outputs1[i, j, 0] = 149
                outputs1[i, j, 1] = 75
                outputs1[i, j, 2] = 76
            elif (outputs[i, j] == 14):
                outputs1[i, j, 0] = 75
                outputs1[i, j, 1] = 149
                outputs1[i, j, 2] = 76
            elif (outputs[i, j] == 15):
                outputs1[i, j, 0] = 73
                outputs1[i, j, 1] = 74
                outputs1[i, j, 2] = 154
            elif (outputs[i, j] == 16):
                outputs1[i, j, 0] = 251
                outputs1[i, j, 1] = 101
                outputs1[i, j, 2] = 102

    for i in range(savenum.shape[0]):
        print(savenum[i,0])
    # outputs1[:,:,i] = outputs[:,:] * (255/16)
    # outputs1 = outputs1.astype(int)
    plt.imshow(outputs1.astype(int))

    plt.show()
    # Cv 的方法
    outputsTemp = outputs1[:,:,0].copy()
    outputs1[:, :, 0] =outputs1[:, :, 2]
    outputs1[:, :, 2] = outputsTemp
    cv2.imwrite("../figure/finalpre" + 'indian' + ".jpg", outputs1)

    # matplotlib方法
    # matplotlib.image.imsave('figure/finalpre.jpeg', np.uint8(outputs1))

    # PIL方法
    # outputs = outputs1.astype(int)
    # # im = Image.fromarray(outputs1)
    # im = Image.fromarray(np.uint8(outputs1))
    #
    #
    # if im.mode == "F":
    #     im = im.convert('I')
    #
    # im = im.convert('RGB')
    # im.save("figure/finalpre.jpg")