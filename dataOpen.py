import cv2
import matplotlib.pyplot as plt
import numpy as np

imgSize = 64
xa, ya, xb, yb = int(0.17*800), int(0.78*800), int(0.48*800), int(0.97*800)
image = "./image/9_14_0.17_0.78_0.48_0.97.png"
elm = cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))

test = np.flip(cv2.resize(elm, (800,800)),1)
test[xa:xb,1-yb:1-ya] = 0
cv2.imshow('test', test)
#cv2.imshow('frame',cv2.resize(elm, (800,800)))
if cv2.waitKey(10000) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
plt.show()