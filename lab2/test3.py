import numpy as np
import cv2
import argparse
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", required=True,help="path to input video")
# args = vars(ap.parse_args())
# print(args["input"])
# img = cv2.imread(args["input"])
img = cv2.imread("p2.jpg")
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
print(image.shape)
height = image.shape[0]
width = image.shape[1]
im = image
print(image[::])
medin = [(0,0)]*9
print(medin)
for i in range(1,height-1):
    for j in range(1,width-1):
        medin[0] = image[i-1,j-1]
        medin[1] = image[i-1,j]
        medin[2] = image[i-1,j+1]
        medin[3] = image[i,j-1]
        medin[4] = image[i,j]
        medin[5] = image[i,j+1]
        medin[6] = image[i+1,j-1]
        medin[7] = image[i+1,j]
        medin[8] = image[i+1,j+1]
        medin = sort(medin)
        print("sortbb",medin)
        im[i][j] = medin[4]


cv2.imshow("medin",im)

cv2.imwrite("p_gay_1.jpg",image)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
