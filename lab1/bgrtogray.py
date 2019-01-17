import numpy as np
import cv2

img = cv2.imread("p2.jpg")

def bgrtogray(image,r,g,b):
    # blue = [0] 7%
    # green = [1] 72%
    # red = [2] 21%
    grayValue = r * image[:,:,2] + g * image[:,:,1] + b * image[:,:,0]
    # convert uint8 to image gray
    gray_img = grayValue.astype(np.uint8)
    return gray_img

image1 = bgrtogray(img,0.299,0.587,0.114)
image2 = bgrtogray(img,0.2126,0.7152,0.0722)
image3 = bgrtogray(img,0.2627,0.6780,0.0593)
# print(image1.shape)
cv2.imshow("GrayScale1",image1)
cv2.imshow("GrayScale2",image2)
cv2.imshow("GrayScale3",image3)
cv2.imwrite('picture_gray1.jpg',image1)
cv2.imwrite('picture_gray2.jpg',image2)
cv2.imwrite('picture_gray3.jpg',image3)
k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
