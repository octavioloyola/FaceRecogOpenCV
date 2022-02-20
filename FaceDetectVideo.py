import cv2
import os
import FaceDetect

# Input
video_cap = cv2.VideoCapture(os.path.abspath("video.mp4"))
# FaceDetection
faceIdent = FaceDetect.FaceIdentification()
# Read the image
while (video_cap.isOpened()):
    # Capture frame-by-frame
    ret, image = video_cap.read()
    faces = faceIdent.extractFaces(image)
    faceIdent.printFaces(faces, image)

    key = cv2.waitKey(1)

    if key == 32:
       cv2.waitKey(-1) 
    elif key == ord('q'):
        break

video_cap.release()
cv2.destroyAllWindows()