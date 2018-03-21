import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
#######################################################quetion1######################################################
# img = cv2.imread('Fry.bmp', 0)
# #a voisin = 1
# def nn_local1(img, threshold):
#     new_mat = np.zeros((img.shape[0], img.shape[1]))
#     res = np.zeros((img.shape[0], img.shape[1]))
#     for i in range(len(new_mat)):
#         for j in range(len(new_mat[0])):
#             for x in range(3):
#                 for y in range(3):
#                     try:
#                         new_mat[i][j] += (img[i-1+x][j-1+y] / 255)
#                     except:
#                         pass
#                 if new_mat[i][j] > threshold:
#                     res[i][j] = 1
#                 else:
#                     res[i][j] = 0
#     return res
#
# #a voisin!=pixel=−1
# def nn_local2(img, threshold, alpha):
#     new_mat = np.zeros((img.shape[0], img.shape[1]))
#     res = np.zeros((img.shape[0], img.shape[1]))
#     for i in range(len(new_mat)):
#         for j in range(len(new_mat[0])):
#             for x in range(3):
#                 for y in range(3):
#                     try:
#                         if x == 1 and y == 1:
#                             new_mat[i][j] += (alpha * img[i][j] / 255)
#                         else:
#                             new_mat[i][j] -= (img[i-1+x][j-1+y] / 255)
#                     except:
#                         pass
#                 if new_mat[i][j] > threshold:
#                     res[i][j] = 1
#                 else:
#                     res[i][j] = 0
#     return res
#
# img_c1 = nn_local1(img, 5)
# cv2.imshow('img_c1', img_c1)
#
# img_c2 = nn_local2(img, 3, 7)
# cv2.imshow('img_c2', img_c2)
# cv2.imshow('img', img)
# cv2.waitKey(0)
#####################################################quetion2######################################################
# def threshold(alpha, val):
#     return 1 / (1 + np.exp(-alpha*val))
#
#     #return 1 if val > 0 else 0
#     # if val > alpha:
#     #     return 1
#     # elif 0 < val:
#     #     return val/alpha
#     # else:
#     #     return 0
#
# def error(M):
#     e =  np.zeros((len(M), len(M[0])))
#     for i in range(len(M)):
#         for j in range(len(M[0])):
#             e[i][j] = (M[i][j] - (i == j))**2
#     return sum(sum(e)), e
#
# def nn3(P, M, s):
#     # temp = np.zeros((len(M), len(M[0])))
#     # for i in range(len(M)):
#     #     for j in range(len(M[0])):
#     #         temp[i][j] = s(1, P[i][j] * M[i][j])
#     MP = M.dot(P)
#     res = np.zeros((len(MP), len(MP[0])))
#     for i in range(len(MP)):
#         for j in range(len(MP[0])):
#             res[i][j] = s(5, MP[i][j])
#     return res
#
# def algo_gene(errors, P):
#     ind = np.argsort(errors)
#     #mutation
#     for i in ind[-5:]:
#         P[i] = 20 * np.random.random((len(M[0]), len(M))) - 10
#     #crossover
#     for i in ind[-15:-5]:
#         a, b = np.random.choice(ind[:3],2)
#         P[i] = (P[a]+P[b])/2
#     #p mutation
#     for i in ind[:2]:
#         for j in range(3):
#             a, b = np.random.randint(0, 36), np.random.randint(0, 6)
#             P[i][a][b] = 2 * np.random.random() - 1
#     #     print(P[i].shape)
#     # print(len(P))
#     return P, P[ind[0]]
#
# names = ['Test1.bmp', 'TestU.bmp', 'TestP.bmp', 'TestO.bmp', 'TestC.bmp', 'TestA.bmp']
# M = []
# for i in names:
#     im = np.array(cv2.imread(i, 0))
#     im = im.flatten()
#     M.append(im/255)
# M = np.array(M)
# P = [20 * np.random.random((len(M[0]), len(M))) - 10 for i in range(20)]
# # errors = [len(M) * len(M[0]), len(M) * len(M[0])-1]
# errors = []
# e = []
# for i in range(2000):
#     errors = []
#     for j in P:
#         MP = nn3(j, M, threshold)
#         ee, e_mat = error(MP)
#         errors.append(ee)
#     print(min(errors))
#     e.append(min(errors))
#     P, best = algo_gene(errors, P)
# #
# print(min(e))
# plt.plot(e)
# plt.show()
#
# test = cv2.imread('Testprime.bmp', 0)
# test = np.array(test.flatten()/255)
# res = test.dot(best)
# for i in range(len(res)):
#     res[i]= threshold(5, res[i])
# print(names[np.argmax(res)],res)
#######################################################question3############################################################3
def threshold(alpha, val):
    #sigmoid
    # return 1 / (1 + np.exp(-alpha*val))

    # #relu
    # return 1 if val > 0 else 0

    if val > alpha*np.exp(-min_e):
        return 1
    elif 0 < val:
        return val/alpha
    else:
        return 0

def error(M):
    e =  np.zeros((len(M), len(M[0])))
    for i in range(len(M)):
        for j in range(len(M[0])):
            e[i][j] = (M[i][j] - (i == j))**2
    return sum(sum(e)), e

def nn4(P, M, s):
    # temp = np.zeros((len(M), len(M[0])))
    # for i in range(len(M)):
    #     for j in range(len(M[0])):
    #         temp[i][j] = s(1, P[i][j] * M[i][j])
    MP = M.dot(P)
    res = np.zeros((len(MP), len(MP[0])))
    for i in range(len(MP)):
        for j in range(len(MP[0])):
            res[i][j] = s(70, MP[i][j])
    return res

def algo_gene(errors, models):
    ind = np.argsort(errors)
    #mutation
    for i in ind[-3:]:
        P[models[i]] = 20 * np.random.random((len(M[0]), len(M))) - 10
        PP[models[i]] = 20 * np.random.random((24,24)) - 10
    #crossover
    for i in ind[3:-3]:
        a, b = np.random.choice(ind[:3], 2)

        P[models[i]] = (P[models[a]] + P[models[b]]) / 2
        PP[models[i]] = (PP[models[a]] + PP[models[b]]) / 2
    #p mutation
    for i in ind[:3]:
        for j in range(30):
            a, b = np.random.randint(0, 36), np.random.randint(0, 6)
            P[models[i]][a][b] += (2 * np.random.random() - 1)
        for j in range(30):
            a, b = np.random.randint(0, 24), np.random.randint(0, 24)
            PP[models[i]][a][b] += (2 * np.random.random() - 1)

    return models[ind[0]]

names = os.listdir('pics')
M = []
for i in names:
    im = np.array(cv2.imread('pics/'+i, 0))
    im = cv2.resize(im,(6,6),interpolation=cv2.INTER_NEAREST)
    im = im.flatten()
    M.append(im/255)
M = np.array(M)
P = [20 * np.random.random((len(M[0]), len(M))) - 10 for i in range(20)]
PP = [20 * np.random.random((24,24)) - 10 for i in range(20)]
# errors = [len(M) * len(M[0]), len(M) * len(M[0])-1]
errors = []
e = []
models = []
backup = ()
min_e=20
for i in tqdm(range(5000)):
    errors = []
    models = []
    for j in range(len(P)):
        MP1 = nn4(P[j], M, threshold)
        MP2 = nn4(PP[j], MP1, threshold)
        ee, e_mat = error(MP2)
        models.append(j)
        errors.append(ee)

    min_e = min(errors)
    print(min_e)
    e.append(min_e)
    index = algo_gene(errors, models)

print(min(e))
plt.plot(e)
plt.show()

# test = cv2.imread('Testprime.bmp', 0)
# test = np.array(test.flatten()/255)
# res = test.dot(best)
# for i in range(len(res)):
#     res[i]= threshold(5, res[i])
# print(names[np.argmax(res)],res)