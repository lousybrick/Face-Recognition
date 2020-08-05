# Read a Video Stream from Camera(Frame by Frame)
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    # grayscale
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    if ret == False:
        continue

    cv2.imshow("Video Frame",frame)
    cv2.imshow("Gray Video Frame",gray_frame)

    # Wait for user input - q : end loop
    key_pressed = cv2.waitKey(1) & 0xFF # converting 32 bit to 8 bit 
    if key_pressed == ord('q'):  # ord() breaks into Ascii value
        break

cap.release()
cv2.destroyAllWindows()