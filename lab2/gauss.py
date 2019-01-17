import numpy as np
import cv2

def bgrtogray(image):
    # blue = [0] 7%
    # green = [1] 72%
    # red = [2] 0.299
    grayValue = 0.299 * image[:,:,2] + 0.587 * image[:,:,1] + 0.114 * image[:,:,0]
    # convert uint8 to image gray
    gray_img = grayValue.astype(np.uint8)
    return gray_img

image = cv2.imread('p2.jpg')
img = bgrtogray(image)
img_out = img.copy()
cv2.imshow("image",image)
cv2.imshow("image_gray",img)
cv2.imwrite("image_gray.jpg",img)
height = img.shape[0]
width = img.shape[1]

# gauss = (1.0/57) * np.array(
#         [[0, 1, 2, 1, 0],
#         [1, 3, 5, 3, 1],
#         [2, 5, 9, 5, 2],
#         [1, 3, 5, 3, 1],
#         [0, 1, 2, 1, 0]])
gauss = (1.0/16) * np.array(
        [[1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]])
sum(sum(gauss))

# for i in np.arange(2, height-2):
#     for j in np.arange(2, width-2):        
#         sum = 0
#         for k in np.arange(-2, 3):
#             for l in np.arange(-2, 3):
#                 a = img.item(i+k, j+l)
#                 p = gauss[2+k, 2+l]
#                 sum = sum + (p * a)
#         b = sum
#         img_out.itemset((i,j), b)
print(img[:])

for i in np.arange(0, height-1):
    for j in np.arange(0, width-1):        
        sum = 0
        for k in np.arange(-1, 2):
            for l in np.arange(-1, 2):
                a = img.item(i+k, j+l)
                p = gauss[1+k, 1+l]
                sum = sum + (p * a)
        b = sum
        img_out.itemset((i,j), b)

cv2.imwrite('images/filter_gauss.jpg', img_out)
cv2.imshow('image_out',img_out)
cv2.waitKey(0)
cv2.destroyAllWindows()