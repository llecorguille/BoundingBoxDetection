import cv2
import numpy as np
import os
import random as r
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

def randomBox(size_x,size_y,size):
    start_box_x = r.randint(0,size_x-size)
    start_box_y = r.randint(0,size_y-size)
    return start_box_x,start_box_y

def mooveBox(x1,y1,x2,y2,length,cpt):
    return int(x1 + (x2-x1)*(cpt/length)), int(y1 + (y2-y1)*(cpt/length))

def drawBox(image_np,x,y,size):
    image_np[x:x+size,y:y+5,:] = [0,255,0]
    image_np[x:x+5,y:y+size,:] = [0,255,0]
    image_np[x+size:x+size+5,y:y+size,:] = [0,255,0]
    image_np[x:x+size,y+size:y+size+5,:] = [0,255,0]

def main(x):
    t = time() + 1
    global maxValue
    cpt = maxValue
    pauseState = True
    boxSize = 300
    print('Pause :', pauseState, 'Press SPACE to start')

    start_box_x, start_box_y = randomBox(np.shape(cap.get_frame())[0],np.shape(cap.get_frame())[1],boxSize)
    end_box_x, end_box_y = randomBox(np.shape(cap.get_frame())[0],np.shape(cap.get_frame())[1],boxSize)
    current_box_x , current_box_y = start_box_x,start_box_y
    j=0

    while cpt <= maxValue + int(sys.argv[1]):
        if time() - t > 0.1 and not(pauseState):
            print('shoot', cpt)
            gray_image = cv2.resize(base_image, (imgSize,imgSize))
            cv2.imwrite('./image/' + str(x) + '_' + str(cpt) + '_' + str(current_box_x) + '_' + str(current_box_y) + '_' + str(current_box_x + boxSize) + '_' + str(current_box_y + boxSize) +'.png', gray_image)
            t = time()
            cpt += 1

        image_np = cap.get_frame()
        base_image = cap.get_frame()
        if(not(pauseState)):
            if(j == 100):
                pauseState = True
                boxSize = r.randint(100,300)
                start_box_x, start_box_y = end_box_x, end_box_y
                end_box_x, end_box_y = randomBox(np.shape(cap.get_frame())[0],np.shape(cap.get_frame())[1],boxSize)
                j=0

            current_box_x , current_box_y = mooveBox(start_box_x,start_box_y,end_box_x,end_box_y,100,j)
            j+=1

        drawBox(image_np,current_box_x, current_box_y,boxSize)
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
    main(x)
