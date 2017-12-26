import glob
import cv2
import random
import numpy as np
import os
import pickle
from PIL import Image
import sys
from threading import Thread, RLock
from time import time

rlock = RLock()
class OpenImage(Thread):
    """ Thread for open images. """
    def __init__(self, listA):
        global data, imgSize
        Thread.__init__(self)
        self.listA = listA
        self.img, self.value, self.size = None, None, None

    def run(self):
        """ Code to execute to open. """
        i = 0
        for elm in self.listA:
            self.value = int(elm.split('\\')[1].split('_')[0])
            if self.value == 0:
              self.img = np.array(cv2.resize(cv2.imread(elm, 0), (imgSize,imgSize)))
              with rlock:
                  data.append([self.img, [-1,-1,-1,-1,-1]])
            else:
              self.img = np.array(cv2.resize(cv2.imread(elm, 0), (imgSize,imgSize)))
              splited = elm.split('\\')[1].split('_')
              self.size = [float(splited[2]),float(splited[3]),float(splited[4]) - float(splited[2]),float(splited[5][:-4]) - float(splited[3]),1.0]
              with rlock:
                  data.append([self.img, self.size])

liste = glob.glob('./image/**')

random.shuffle(liste)
#pourcentage d'exemples pour train le modèle
#pourcentage pour le test 1 - split
split = 0.90
nbClass = 4
imgSize = 64

data = []
#Chargement en RAM des images trouvées
# Threads Creation
t1 = time()
threads = []

nbThread = 20
size = int(len(liste)/nbThread)
for x in range(nbThread):
    threads.append(OpenImage(liste[x*size:(x+1)*size]))

# Lancement des threads
for thread in threads:
    thread.start()


# Attend que les threads se terminent
for thread in threads:
    thread.join()

print('len de data', len(data), time() - t1)

print('Chargement en RAM des images done ...')
#Traitement des images pour l'entrainement du modèle
X_train = []
y_train = []
data_train = []
for elm in data[:int(len(data)*split)]:
    boundingBox = [elm[1][0], 1-elm[1][3], elm[1][2], 1-elm[1][1],elm[1][4]]
    data_train.append([np.flip(elm[0],1), boundingBox])
    data_train.append([elm[0], elm[1]])

print('Traitement data_train done ...')
#Traitement des images pour le test du modèle
X_test = []
y_test = []
data_test = []
for elm in data[int(len(data)*split):]:
    boundingBox = [elm[1][0], 1-elm[1][3], elm[1][2], 1-elm[1][1],elm[1][4]]
    data_test.append([np.flip(elm[0],1), boundingBox])
    data_test.append([elm[0], elm[1]])

print('Traitement data_test done ...')
data = 0
random.shuffle(data_test)
random.shuffle(data_train)


for elm in data_train:
    X_train.append(elm[0])
    y_train.append(elm[1])
data_train = 0

for elm in data_test:
    X_test.append(elm[0])
    y_test.append(elm[1])
data_test = 0

X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
print('Ready to dump')

save_dir = './dataTrain/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)



np.save('./dataTrain/Xtest_0', X_test)
print("Nombres exemples de test", len(X_test))
X_test = 0
np.save('./dataTrain/Ytest_0', y_test)
y_test = 0

np.save('./dataTrain/Ytrain_0', y_train)
y_train = 0
np.save('./dataTrain/Xtrain_0', X_train)
print("Nombres exemples d'entrainement", len(X_train))
X_train = 0





