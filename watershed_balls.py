import cv2
import numpy as np

# red color
color = [np.array([0, 120, 120]), np.array([10, 255, 255])]

cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, frame = cam.read()
    frame = frame[:,::-1]
    blurred = cv2.medianBlur(frame, 25)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    mask = np.array(cv2.inRange(hsv, color[0], color[1]))
    
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    ret, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)

    fg = np.uint8(fg)
    confuse = cv2.subtract(mask, fg)

    ret, markers = cv2.connectedComponents(fg)
    markers += 1
    markers[confuse == 255] = 0

    wmarkers = cv2.watershed(frame, markers.copy())

    print(f'Balls: {np.max(wmarkers) - 1}')

    cv2.imshow('Camera', mask)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()