import numpy as np
import cv2
import argparse
img = cv2.imread("rack.png")
cv2.imshow("img",img)
def bgrtogray(image):
    # blue = [0] 0.114 # green = [1] 0.587 # red = [2] 0.299
    grayValue = 0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]
    # convert uint8 to image gray
    gray_img = grayValue.astype(np.uint8)
    
    return gray_img

image = bgrtogray(img)
cv2.imshow("GrayScale1",image)
(height,width) = image.shape
im = image
for i in range(0,height-1):
    for j in range(0,width-1):
        convolutional = []
        for k in range(-1, 2):
            for l in range(-1, 2):
                pix = image.item(i+k, j+l)
                convolutional.append(pix)
        mean = sum(convolutional)/9
        im[i][j] = int(mean)

cv2.imwrite("p_gay_1.jpg",image)
cv2.imshow("mean",im)
cv2.imwrite("mean.jpg",im)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
