import numpy as np
import cv2
import argparse
img = cv2.imread("p3.jpg")
cv2.imshow("img",img)
def bgrtogray(image):
    # blue = [0] 7%
    # green = [1] 72%
    # red = [2] 0.299
    grayValue = 0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]
    # convert uint8 to image gray
    gray_img = grayValue.astype(np.uint8)    
    return gray_img

def sort(list):
    for i in range(len(list)):
        m = i
        for j in range(i+1,len(list)):
            if list[m] > list[j]:
                m = j
        list[i], list[m] = list[m] , list[i]
        return list

image = bgrtogray(img)
cv2.imshow("GrayScale1",image)
# print(image.shape)
(height,width) = image.shape
im = image
# print(image[::])
# print(medin)
for i in range(0,height-1):
    for j in range(0,width-1):
        neighbors = []
        for k in range(-1, 2):
            for l in range(-1, 2):
                a = image.item(i+k, j+l)
                neighbors.append(a)
        neighbors = sorted(neighbors)
        im[i][j] = neighbors[4]
cv2.imshow("medin",im)
cv2.imwrite("p_gay_1.jpg",image)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
