import matplotlib.pyplot as plt
import numpy as np
import cv2
global pixel 
pixel = np.zeros(256)
image = cv2.imread('p1.jpg')
cv2.imshow("ori",image)

def bgrtogray(image):
    # blue = [0] 7%# green = [1] 72%# red = [2] 0.299
    grayValue = 0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]
    # convert uint8 to image gray
    gray_img = grayValue.astype(np.uint8)
    return gray_img

def backwhile(image):
    (height,width) = image.shape
    img_out = image
    input1 = int(input("input : "))
    for i in range(0,height-1):
        for j in range(0,width-1):
            # print(image[i,j])
            pixel[image[i,j]] += 1 
            if image[i,j] > input1:
                img_out[i,j] = 255
            else:
                img_out[i,j] = 0

    return img_out

img = bgrtogray(image)
cv2.imshow("gray",img)
cv2.imwrite("gray.jpg",img)

bw = backwhile(img)
cv2.imshow("blackwhile",bw)
cv2.imwrite("blackwhile.jpg",bw)

num = range(0,255)
print(pixel)
plt.plot(pixel)
fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
axs[0].hist(pixel, bins=20)
# plt.axis([0, 256, 0, 500])
plt.grid(True)

# plt.plot(pixel)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()