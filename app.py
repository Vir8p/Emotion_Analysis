from cv2 import IMREAD_REDUCED_GRAYSCALE_2
from moviepy.editor import *
import os
from deepface import DeepFace
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import emtn

video_path = 'videotest.mp4'
dur = VideoFileClip(video_path).duration

no_of_clip = int((dur//10) if dur%10==0 else 1+(dur//10))
#print(no_of_clip)
'''
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
'''

def val(s,d):
    if s not in d:
        return 0
    else:
        return d[s]

def barplot(labels,angry,disgust,fear,happy,neutral,sad,surprise):

    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - 0.3, angry, width, label='Angry')
    rects2 = ax.bar(x - 0.2, disgust, width, label='Disgust')
    rects3 = ax.bar(x - 0.1, fear, width, label='Fear')
    rects4 = ax.bar(x , happy, width, label='Happy')
    rects5 = ax.bar(x + 0.1, neutral, width, label='Neutral')
    rects6 = ax.bar(x + 0.2, sad, width, label='Sad')
    rects7 = ax.bar(x + 0.3, surprise, width, label='Surprise')


    ax.set_ylabel('Accuracy in %')
    ax.set_title('Emotions of every 10 sec')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    ax.bar_label(rects5, padding=3)
    ax.bar_label(rects6, padding=3)
    ax.bar_label(rects7, padding=3)

    fig.tight_layout()

    plt.show()

def discplot(angry,disgust,fear,happy,neutral,sad,surprise):
    category_names = ['Angry', 'Disgust','Fear', 'Happy', 'Neutral','Sad','Surprise']
    zz=0
    results={}
    for a in range(len(angry)):
        results[f'{zz*10}-{(zz+1)*10} sec']= [angry[a],disgust[a],fear[a],happy[a],neutral[a],sad[a],surprise[a]]
        zz+=1

    def survey(results, category_names):
        """
        Parameters
        ----------
        results : dict
            A mapping from question labels to a list of answers per category.
            It is assumed all lists contain the same number of entries and that
            it matches the length of *category_names*.
        category_names : list of str
            The category labels.
        """
        labels = list(results.keys())
        data = np.array(list(results.values()))
        data_cum = data.cumsum(axis=1)
        category_colors = plt.colormaps['RdYlGn'](
            np.linspace(0.15, 0.85, data.shape[1]))

        fig, ax = plt.subplots(figsize=(9.2, 5))
        ax.invert_yaxis()
        ax.xaxis.set_visible(False)
        ax.set_xlim(0, np.sum(data, axis=1).max())

        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths
            rects = ax.barh(labels, widths, left=starts, height=0.5,
                            label=colname, color=color)

            r, g, b, _ = color
            text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
            ax.bar_label(rects, label_type='center', color=text_color)
        ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
                loc='lower left', fontsize='small')

        return fig, ax

    survey(results, category_names)
    plt.show()

clip=[]
labels=[]
angry,disgust,fear,happy,neutral,sad,surprise=[],[],[],[],[],[],[]
emotion={}
os.mkdir('Clips')
for i in range(no_of_clip):
    
    if (i+1)*10 < dur:
        vid=(VideoFileClip(video_path).subclip(i*10,(i+1)*10))
    else:
        vid=(VideoFileClip(video_path).subclip(i*10,dur))
    
    vid.write_videofile(f"Clips/clip{i+1}.mp4")
    
    clip.append(f"Clips/clip{i+1}.mp4")
    labels.append(f'{i*10}-{(i+1)*10}')


emotion,angry,disgust,fear,happy,neutral,sad,surprise=emtn.main(clip,angry,disgust,fear,happy,neutral,sad,surprise)
'''
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
'''

print(emotion)

barplot(labels,angry,disgust,fear,happy,neutral,sad,surprise)  # BAR Plot

discplot(angry,disgust,fear,happy,neutral,sad,surprise)        # DISCRETE Plot
 
for i in clip:
    VideoFileClip(i).close()
    print(i)


for i in range(no_of_clip):
    os.remove(f"Clips/clip{i+1}.mp4")
os.rmdir('Clips')
