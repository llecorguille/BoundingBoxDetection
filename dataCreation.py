import cv2
import numpy as np
import os
import random as r
import copy
from time import time, sleep
import glob

import sys

class VideoCamera(object):
    def __init__(self, index=0):
        self.video = cv2.VideoCapture(index)
        self.index = index
        print(self.video.isOpened())

    def __del__(self):
        self.video.release()

    def get_frame(self, in_grayscale=False):
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

#info sur le geste à faire
#recup auto du numero de l'image à ajouter 0_{nb}
#save in color or in gray

# Open a new thread to manage the external cv2 interaction
cv2.startWindowThread()
cap = VideoCamera()

imgSize = 256
cameraSize = (800, 600)
nbClass = 15

# 0 => rien
# 1 => Poing fermé
# 2 => pouce haut
# 3 => pouce bas
# 4 => main ouverte doigts serrés
# 5 => main ouverte doigts ecartés
# 6 => pouce + auriculaire horizontal
# 7 => pouce + auriculaire vertical
# 8 => victoire
# 9 => C
# 10 => ok
# 11 => victoire serré
# 12 => victoire serré horizontal
# 13 => rock and roll
# 14 => rock and roll horizontal

gestures = ['None', 'fist', 'thumb up', 'thumb down', \
            'stop', 'catch', 'swing', 'phone', 'victory', \
            'C', 'okay', '2 fingers', '2 fingers horiz', \
            'rock&roll', 'rock&roll horiz']

ratio = [0,1,0.7,0.7,0.7,1,1.5,1,0.7,0.8,0.8,0.5,2,0.7,1.5]

def randomBox(size_x,size_y,size,ratio):
    start_box_x1 = r.randint(0,size_x-size)
    start_box_y1 = r.randint(0,size_y-int(size*ratio))
    return start_box_x1,start_box_y1

def mooveBox(x1,y1,x2,y2,length,cpt):
    return int(x1 + (x2-x1)*(cpt/length)), int(y1 + (y2-y1)*(cpt/length))

def getCoordBox(x1,y1,size,ratio):
    #returns x2,y2
    '''if ratio>1:
        x2 = x1 + int(size*ratio)
        y2 = y1 + size
    else:'''

    x2 = x1 + size
    y2 = y1 + int(size*ratio)
    return x2,y2

def drawBox(image_np,x1,y1,x2,y2):
    image_np[x1:x2,y1:y1+5,:] = [0,255,0]
    image_np[x1:x1+5,y1:y2,:] = [0,255,0]
    image_np[x2:x2+5,y1:y2,:] = [0,255,0]
    image_np[x1:x2,y2:y2+5,:] = [0,255,0]

def main(x,ratio):
    t = time() + 1
    global maxValue
    cpt = maxValue
    pauseState = True
    boxSize = 150
    print('Pause :', pauseState, 'Press SPACE to start')

    image_np = cap.get_frame()
    start_box_x1, start_box_y1 = randomBox(image_np.shape[0],image_np.shape[1],boxSize,ratio)
    end_box_x1, end_box_y1 = randomBox(image_np.shape[0],image_np.shape[1],boxSize,ratio)
    current_box_x1 , current_box_y1 = start_box_x1,start_box_y1
    current_box_x2 , current_box_y2 = getCoordBox(current_box_x1,current_box_y1,boxSize,ratio)
    j=0

    while cpt <= maxValue + int(sys.argv[1]):
        if time() - t > 0.1 and not(pauseState):
            print('shoot', cpt)
            gray_image = cv2.resize(base_image, (imgSize,imgSize))
            if(ratio != 0):
                xa, xb = int(current_box_x1/image_np.shape[0]*100)/100, int(current_box_x2/image_np.shape[0]*100)/100
                ya, yb = int(current_box_y1/image_np.shape[1]*100)/100, int(current_box_y2/image_np.shape[1]*100)/100
                cv2.imwrite('./image/' + str(x) + '_' + str(cpt) + '_' + str(xa) + '_' + str(ya) + '_' + str(xb) + '_' + str(yb) +'.png', gray_image)
            else:
                cv2.imwrite('./image/' + str(x) + '_' + str(cpt) + '.png', gray_image)
            t = time()
            cpt += 1

        image_np = cap.get_frame()
        base_image = copy.deepcopy(image_np)
        if(not(pauseState)):
            if(j == 100):
                pauseState = True
                boxSize = r.randint(100,200)
                start_box_x1, start_box_y1 = end_box_x1, end_box_y1
                end_box_x1, end_box_y1 = randomBox(image_np.shape[0],image_np.shape[1],boxSize,ratio)
                ratio = ratio + ((r.random()-0.5)*0.2)
                j=0

            current_box_x1 , current_box_y1 = mooveBox(start_box_x1,start_box_y1,end_box_x1,end_box_y1,100,j)
            current_box_x2 , current_box_y2 = getCoordBox(current_box_x1,current_box_y1,boxSize,ratio)
            j+=1

        if(ratio != 0):
            drawBox(image_np,current_box_x1, current_box_y1,current_box_x2,current_box_y2)
        cv2.imshow('object detection', cv2.resize(np.flip(image_np,1), cameraSize))

        key = cv2.waitKey(25) & 0xFF
        if key == ord(' '):
            pauseState = not(pauseState)
            print('Pause :', pauseState, 'Press SPACE to change state')
        elif key == ord('q'):
            cv2.destroyAllWindows()
            break

save_dir = 'image/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


""" Récupération du dernier index d'image dans le dossier des images """
liste = glob.glob(save_dir + '*.png')
maxValue = -1
for elm in liste:
    value = int(elm.split('_')[1].split('.')[0])
    if value > maxValue:
        maxValue = value

for x in range(nbClass):
    print('Lancement main :', gestures[x])
    main(x,ratio[x])
