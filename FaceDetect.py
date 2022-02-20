import cv2
import sys
import os


class FaceIdentification:
    def __init__(self):
        # filePath of the frontal face classifier
        face_classifier = os.path.abspath("haarcascade_frontalface_default.xml")
        # Create the haar cascade
        self.faceCascade = cv2.CascadeClassifier(face_classifier)

    def extractFaces(self, image):
        # Convert to gray to faster processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Detect faces in the image
        faces = self.faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.25,
            minNeighbors=5,
            minSize=(30, 30)
        ) 
        return faces
    
    def printFaces(self, faces, image):
        # Text about faces found
        text = "Faces found: "+ str(len(faces))
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0,255,0), 2)
            # plotting bounding ellipse  
            #center = (x + w//2, y + h//2)
            #cv2.ellipse(image, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
        cv2.putText(image, text, (300,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        cv2.imshow("Face Detection", image)

if __name__ == '__main__':
    faceIdent = FaceIdentification()
    image = cv2.imread(sys.argv[1])
    faces = faceIdent.extractFaces(image)
    faceIdent.printFaces(faces, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()