import cv2
import matplotlib.pyplot as plt

imgSize = 64
xa, ya, xb, yb = int(0.36*800), int(0.5*800), int(0.68*800), int(0.74*800)
image = "./image/5_0_0.36_0.5_0.68_0.74.png"
elm = cv2.resize(cv2.imread(image, 0), (imgSize,imgSize))

test = cv2.resize(elm, (800,800))
img = test[xa:xb,ya:yb]
cv2.imshow('test', img)
#cv2.imshow('frame',cv2.resize(elm, (800,800)))
if cv2.waitKey(10000) & 0xFF == ord('q'):
      cv2.destroyAllWindows()
plt.show()