import cv2

# BGR to BGR display
img = cv2.imread('dog.png')
gray = cv2.imread('dog.png',cv2.IMREAD_GRAYSCALE)
cv2.imshow('Dog Image',img)
cv2.imshow('Gray Dog Image',gray)

cv2.waitKey(0) # 0 = wait for infinite amount of time
cv2.destroyAllWindows()