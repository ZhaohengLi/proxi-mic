# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

# import env
import librosa

fake_label:np.ndarray=None
threshold=None

class Detection():
    def __init__(self):
        self.min_threshold=0.3
        self.threshold=self.min_threshold
        self.min_t2=0.9995   # d^2y/dt^2
        self.t2=self.min_t2

    def reset(self):
        self.threshold=self.min_threshold

    def run(self, wav:np.ndarray)->bool: 
        """
        detect algorithm
        """
        #fast detect
        if wav.max()>self.threshold:
            result=True
        else:
            result=False
        if result==True: 
            result2=self._solve(wav)
            if result2==False:
                self.threshold+=0.01
                self.t2+=(1-self.min_t2)/20
        self.threshold=max(self.min_threshold,self.threshold*self.t2)
        self.t2=max(self.min_t2,self.t2*0.9999998)
        global threshold
        threshold.append(self.threshold)
        return result
    
    def _solve(self, wav:np.ndarray)->bool:
        """
        more effective  but slower detection
        """
        # fake label
        global fake_label
        return ground_truth(fake_label)

def ground_truth(label)->bool:
    return label.sum()*10>len(label)

def draw_episode(episode):
    wav,tags,label=episode
    print('draw')
    librosa.output.write_wav("./test_episode.wav",wav,22050)
    p=([[],[]],[[],[]])
    point_num=10000
    ds=len(wav)//point_num
    for x in range(0,len(wav),ds):
        y_max=wav[x:x+ds].max()
        y_min=wav[x:x+ds].min()
        y_x=wav[x]
        for y in (y_max,y_min,y_x):
            p[label[x]][0].append(x)
            p[label[x]][1].append(y)
    x1,y1,x2,y2=(p[0][0],p[0][1],p[1][0],p[1][1])
    size=2
    plt.figure(figsize=(15, 5))
    plt.scatter(x1,y1,c='b',s=size)
    plt.scatter(x2,y2,c='r',s=size)
    

def test_episode(episode):
    """
    fast algorithm test
    plot the wav data and threshold curve
    """
    D=Detection()
    size=2048
    wav,tags,label=episode
    draw_episode(episode)
    right=wrong=0
    actarr=np.zeros(len(wav),dtype=np.bool) # if triger then fill 2048 True
    global threshold,fake_label
    threshold=[]
    summary=np.array([[0,0],[0,0]],dtype=np.int32)
    for i in range(0,len(wav),size):
        fake_label=label[i:i+size]
        activate=D.run(wav[i:i+size])
        if activate:
            actarr[i:i+size]=True
        gt=ground_truth(fake_label)
        if gt==False:
            summary[int(gt),int(activate)]+=1
        if activate==gt:
            right+=1
        else:
            wrong+=1
    for tag in tags:
        r = actarr[tag:tag+size].any()
        summary[int(True),int(r)]+=1
    print(right,wrong)
    print(summary)
    plt.plot([i*size for i in range(len(threshold))],threshold,c='g')
    plt.show()

if __name__ == '__main__':
    # test algorithm
    # e=env.Environment()
    # while(True):
    #     test_episode(e.genEpisode(P=0.5))
    file_name = "./user_data/env/20200428_102528.m4a"
    stereo, sr = librosa.load(file_name, sr=8000, mono=False)
    left = np.sum(stereo[0]**2)
    right = np.sum(stereo[1]**2)
    if left > right:
        y = stereo.copy()[0]
    else:
        y = stereo.copy()[1]
    test_episode(y)
