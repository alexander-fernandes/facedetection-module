#
# a simple python program using the mediapipe and opencv 
# library frameworks to develop a stylized face detection 
# module based around Machine Learning.
#

# import libraries
import cv2
import mediapipe as mp
import time

# FaceDetector class
class FaceDetector():

    def __init__(self, minDetectionConfidence=0.5):
        self.minDetectionConfidence = minDetectionConfidence

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.minDetectionConfidence)

    def locate_faces(self, frame, draw=True):

        # this is the class function responsible for using the 
        # mediapipe framework to detect faces. we only need to order
        # them in bounding boxes and ship them off to self.fancy_draw.

        # bounding boxes simply contain points for the minimum dimension
        # of a probable detected object.

        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # we need to convert any stream into RGB
        self.results = self.faceDetection.process(imgRGB)

        bboxs = [] # list to store all the future bounding boxes, more lists.

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):

                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)  
                bboxs.append([id, bbox, detection.score])

                if draw:
                    # draw the object detection box
                    img = self.fancy_draw(frame, bbox)
                    # a detection score as percentage to know how accurate our system is
                    cv2.putText(frame, f"{int(detection.score[0]*100)}%", (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1)

        return frame, bboxs
    
    # lets place a box on that face?
    def fancy_draw(self, img, bbox, l=30, t=3, rt=1):
        # Draws in a box around the perimeter of every 
        # face detected on the stream    

        x, y, w, h = bbox
        x1, y1 = x+h, y+h
        
        cv2.rectangle(img, bbox, (255,0,255), rt) # (255,0,255)

        # just to make it look cooler, drawing thicker corners

        # top left x, y
        cv2.line(img, (x, y), (x + l, y), (0,255,0), t)
        cv2.line(img, (x, y), (x, y + l), (0,255,0), t)

        # top right x1, y
        cv2.line(img, (x1, y), (x1 - l, y), (0,255,0), t)
        cv2.line(img, (x1, y), (x1, y + l), (0,255,0), t)

        # bottom left x, y1
        cv2.line(img, (x, y1), (x + l, y1), (0,255,0), t)
        cv2.line(img, (x, y1), (x, y1 - l), (0,255,0), t)

        # bottom right x1, y1
        cv2.line(img, (x1, y1), (x1 - l, y1), (0,255,0), t)
        cv2.line(img, (x1, y1), (x1, y1 - l), (0,255,0), t)

        return img

def main():

    video_capture = cv2.VideoCapture(0)
    #video_capture = cv2.VideoCapture("/path/to/video")
    pTime = 0
    detector = FaceDetector(0.7)
    while True:

        success, frame = video_capture.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        frame, bboxs = detector.locate_faces(frame)

        try:
            cTime = time.time()
            fps = 1/(cTime - pTime)
            pTime = cTime
        # because i couldn't find a better way
        # of dealing with my FPS. kept crashing.
        except ZeroDivisionError: 
            pass

        cv2.putText(frame, f"FPS: {int(fps)}", (20,30), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 1)
        cv2.imshow('Video', frame)

        # as in, press 'q' when done
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()