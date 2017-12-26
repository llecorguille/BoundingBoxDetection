import tensorflow as tf
import glob
import cv2
import random as r
import numpy as np
import os
import ctypes
import time

def drawBox(image_np,x1,y1,x2,y2):
    image_np[x1:x2,y1:y1+5,:] = [0,255,0]
    image_np[x1:x1+5,y1:y2,:] = [0,255,0]
    image_np[x2:x2+5,y1:y2,:] = [0,255,0]
    image_np[x1:x2,y2:y2+5,:] = [0,255,0]

new_saver = tf.train.import_meta_graph('./final_model_bis/best_model.meta')
graph = tf.get_default_graph()

x = graph.get_tensor_by_name("input_x:0")
y = graph.get_tensor_by_name("labels:0")
keep_prob = graph.get_tensor_by_name("dropRate:0")
y_pred = graph.get_tensor_by_name("predictionBox:0")


liste = glob.glob('image/**')
imgSize = 64

pasMain = False
cap = cv2.VideoCapture(0)
t = time.time()
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  new_saver.restore(sess, tf.train.latest_checkpoint('./final_model_bis/'))
  while True:
    ret, image_np = cap.read()
    """elm = liste[r.randint(0,len(liste)-1)]
    image_np = cv2.imread(elm)"""
    
    gray_image = cv2.cvtColor(cv2.resize(image_np, (imgSize,imgSize)), cv2.COLOR_BGR2GRAY)
    t2 = time.time()
    gray_image = cv2.equalizeHist(gray_image)
    result = y_pred.eval({x:[gray_image], keep_prob: 1})[0]

    
    xa, ya, xb, yb, confidence = int(result[0]*300), int(result[1]*400), int(result[2]*300), int(result[3]*400), result[4]
    image_np = cv2.resize(image_np, (400,300))
    if xa > 0 and xb > 0 and ya > 0 and yb > 0:
      if confidence > 0.5:
        drawBox(image_np,xa,ya,xa+xb,ya+yb)
        print('Box confidence',confidence)
      print(result, 1/(time.time() - t), 1/(time.time() - t2))
    else:
      if confidence < -0.5:
        print('Pas de main', confidence)
        pasMain = True
    cv2.imshow('object detection', image_np)
    if pasMain:
      input('Pause')
      pasMain = False
    t = time.time()
    if cv2.waitKey(50) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
      break