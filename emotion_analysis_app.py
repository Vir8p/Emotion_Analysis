from cv2 import IMREAD_REDUCED_GRAYSCALE_2
from moviepy.editor import *
import os
from deepface import DeepFace
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import emtn

video_path =         'confident-happy.webm'    # "parshawan-harnoor.mp4"   'sad-worried.webm'  

dur = VideoFileClip(video_path).duration
#ats = analysis_timeslot = 1 

if dur >=40 :
    ats =10
elif dur>25:
    ats=5
elif dur>12:
    ats=2
else:
    ats=1


no_of_clip = int((dur//ats) if dur%ats==0 else 1+(dur//ats))
#print(no_of_clip)

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

def lineplot(labels,angry,disgust,fear,happy,neutral,sad,surprise):
    
    plt.plot(labels, angry, label = "Angry", marker = '*', markersize = 6)              
    plt.plot(labels, disgust, label = "Disgust", marker = '*', markersize = 6)         
    plt.plot(labels, fear, label = "Fear", marker = '*', markersize = 6)           
    plt.plot(labels, happy, label = "Happy",marker = '*', markersize = 6)       
    plt.plot(labels, neutral, label = "Neutral", marker = '*', markersize = 6)         
    plt.plot(labels, sad, label = "Sad", marker = '*', markersize = 6)           
    plt.plot(labels, surprise, label = "Surprise", marker = '*', markersize = 6)       
    
    #leg = plt.legend(loc='upper right')
    plt.legend(bbox_to_anchor =( 0.9,1.15), ncol = 4)
    plt.xlabel("Duration")
    plt.ylabel("% Correct")
    plt.title('Emotion Analysis:',loc='left')
    plt.show()


clip=[]
labels=[]
angry,disgust,fear,happy,neutral,sad,surprise=[],[],[],[],[],[],[]
emotion={}
os.mkdir('Clips')
for i in range(no_of_clip):
    
    if (i+1)*ats < dur:
        vid=(VideoFileClip(video_path).subclip(i*ats,(i+1)*ats))
    else:
        vid=(VideoFileClip(video_path).subclip(i*ats,dur))
    
    vid.write_videofile(f"Clips/clip{i+1}.mp4")
    
    clip.append(f"Clips/clip{i+1}.mp4")
    labels.append(f'{i*ats}-{(i+1)*ats}')


emotion,angry,disgust,fear,happy,neutral,sad,surprise=emtn.main(clip,angry,disgust,fear,happy,neutral,sad,surprise)

print(emotion)

lineplot(labels,angry,disgust,fear,happy,neutral,sad,surprise)  # Line plot

barplot(labels,angry,disgust,fear,happy,neutral,sad,surprise)  # BAR Plot

#discplot(angry,disgust,fear,happy,neutral,sad,surprise)        # DISCRETE Plot
 
for i in clip:
    VideoFileClip(i).close()
    print(i)

for i in range(no_of_clip):
    os.remove(f"Clips/clip{i+1}.mp4")
os.rmdir('Clips')
