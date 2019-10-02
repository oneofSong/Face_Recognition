#box': [880, 646, 48, 61
#968, 428, 41, 52

import cv2

frame = cv2.imread("/home/bit/Downloads/twice.jpg", cv2.IMREAD_COLOR)
frame = cv2.rectangle(frame, (abs(958),abs(428)), (abs(41),abs(52)), (0, 255, 0), 2)
frame = cv2.putText(frame, "NAME", (abs(41),abs(52)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
cv2.imshow('aa', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()