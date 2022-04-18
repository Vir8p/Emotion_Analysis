from cv2 import IMREAD_REDUCED_GRAYSCALE_2
from moviepy.editor import *
import os
from deepface import DeepFace
import cv2
import mediapipe as mp

class FaceDetector():
    def __init__(self,minDetectionCon=0.75): 

        self.minDetectionCon=minDetectionCon
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.facedetection=self.mp_face_detection.FaceDetection(0.75)

    def findFaces(self,img,draw=True):

        imgRGB= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.facedetection.process(img)
        #print(self.results)
        bboxs = []
        emot = []
        if self.results.detections:
            for id , detection in enumerate(self.results.detections):
                #print(id,detection)
                #print(detection.score)
                #print(detection.location_data.relative_bounding_box)
                #mp_drawing.draw_detection(img,detection)

                bboxC=detection.location_data.relative_bounding_box
                ih, iw, ic=img.shape
                bbox=int(bboxC.xmin * iw),int(bboxC.ymin * ih), int(bboxC.width * iw),int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])

                if draw:    
                    try:
                        obj = DeepFace.analyze(img, actions = ['emotion'])
                        #print(obj)

                        s=obj['dominant_emotion']
                        emot.append(s)
                        emot.append(round(obj['emotion'][s],1))

                        #print(obj['dominant_emotion'], obj['emotion'][s])

                        cv2.putText(img, f'{s}', (bbox[0],bbox[1]+bbox[3]+22), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
                    except:
                        #print('No Face')
                        cv2.putText(img, f'Not_Detected', (bbox[0],bbox[1]+bbox[3]+22), cv2.FONT_HERSHEY_PLAIN, 2, (0,2,255), 1)

                    cv2.putText(img, f'{int(detection.score[0]*100)}% Correct', (bbox[0],bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
        else:
            cv2.putText(img, f'No_Face / Face_Not_Recognized', (20,70), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)

        return img, bboxs ,emot

def val(s,d):
    if s not in d:
        return 0
    else:
        return d[s]
emotion={}
def main(clip,angry,disgust,fear,happy,neutral,sad,surprise):
    x=0
    for i in clip:
        cap=cv2.VideoCapture(i)
        emot={}
        flag=0
        
        f_detector= FaceDetector()
        
        try:
            while True:
                success, img = cap.read()
                
                if not success:
                    break
                
                #img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                imgr,bbox, emo=f_detector.findFaces(img, draw= True)
                #print(bbox)
                
                if emo!=[]:
                    if emo[0] not in emot:
                        emot[emo[0]]=emo[1]
                    else:
                        if emo[1]> emot[emo[0]]:
                            emot[emo[0]]=emo[1]
                
                cv2.imshow("Camera:",imgr )
                
                if cv2.waitKey(1) & 0xFF==27:
                    flag=1
                    break
            imgr.release()
            img.release()
        except AttributeError:
            pass
        finally:
            print(i,emot)
            emotion[f'{x*10}-{(x+1)*10}']=emot
            x+=1

            angry.append(val('angry',emot))
            disgust.append(val('disgust',emot))
            fear.append(val('fear',emot))
            happy.append(val('happy',emot))
            neutral.append(val('neutral',emot))
            sad.append(val('sad',emot))
            surprise.append(val('surprise',emot)) 

        if flag:
            break
    cv2.destroyAllWindows()
    return emotion,angry,disgust,fear,happy,neutral,sad,surprise


if __name__=='__main__':
    main()