import cv2
import numpy as np

src = r'examples/saad2.jpg'

img = cv2.imread(src)
print(img.shape)

img_resize = cv2.resize(img,(256,256))
cv2.imwrite('examples/saad_resize2.jpg',img_resize)