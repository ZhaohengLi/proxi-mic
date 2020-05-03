#-*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display


#plot parameters
#https://matplotlib.org/examples/color/colormaps_reference.html
cm=plt.cm.get_cmap('jet')

def drawSpec(path:str,S:np.ndarray):
    assert(type(path)==str)
    plt.gcf().set_size_inches(15,5)
    librosa.display.specshow(np.array(S), y_axis='mel',sr=8000,hop_length=40,cmap=cm)
    plt.gcf().savefig(path)
    plt.show()

# CnnNet5
class Cnn(nn.Module):
    def __init__(self, in_channels=20, out_channels=2):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=100, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(num_features=100)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(
            in_channels=100, out_channels=500, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(num_features=500)
        self.pool2 = lambda x: torch.max(x, 2)[0]
        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(100, out_channels)

    def forward(self, x):
        x=F.relu(self.batchnorm1(self.conv1(x)))
        print(x.shape)
        drawSpec('conv1.png',x[0].cpu().numpy())
        x = self.pool1(x)
        print(x.shape)
        drawSpec('pool1.png',x[0].cpu().numpy())
        x = F.relu(self.batchnorm2(self.conv2(x)))
        print(x.shape)
        drawSpec('conv2.png',x[0].cpu().numpy())
        x = self.pool2(x)
        print(x.shape)
        x = x.view(-1, 500)
        x = F.relu(self.fc1(x))
        print(x.shape)
        x = self.fc2(x)
        print(x.shape)
        return x

net=Cnn(in_channels=20, out_channels=2)
net.load_state_dict(torch.load('./speech.model'))
assert(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net=net.to(device)
net.eval()

def FilterBankFeature(wav: np.ndarray, sr: int, n_feature=20) -> torch.tensor:
    n_fft = sr // 50        # STFT window size
    hop_length = n_fft // 4  # STFT hop length

    from functools import lru_cache
    @lru_cache(maxsize=None)
    def FilterBank(sr, n_fft, n_feature):
        # <250Hz all bins, >=250Hz mel_freq
        df = float(sr/2)/(int(1 + n_fft//2)-1)
        low_bins = round(250/df)
        up_bins = n_feature-low_bins
        weight = np.zeros((n_feature, int(1 + n_fft//2)), dtype=np.float32)
        for i in range(low_bins):
            weight[i, i+1] = 1.0/df
        weight[low_bins:, :] = librosa.filters.mel(
            sr=sr, n_fft=n_fft, n_mels=up_bins, fmin=low_bins*df, htk=True)
        
        print(weight.shape,'filter-bank shape')
        plt.gcf().set_size_inches(5,15)
        plt.plot(weight.T)
        plt.gcf().savefig('filter-bank.png')
        plt.show()

        return weight
    S = librosa.amplitude_to_db(
        np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))
    print(S.shape,'stft shape')
    drawSpec('stft.png',S)
    
    S = np.dot(FilterBank(sr, n_fft, n_feature=20), S)
    #Create a Filterbank matrix to combine FFT bins
    result = torch.from_numpy(S)
    return result

def run(y)->int:
    assert(len(y)==8000)
    F=FilterBankFeature(y, sr=8000)
    drawSpec('nn_input.png',F)
    F=torch.unsqueeze(F,0)
    F=F.to(device)
    with torch.no_grad():
        outputs = net(F)
    _, predicted = torch.max(outputs, 1)
    predicted=predicted[0].item()
    return predicted

def plotCNNFrameWork():
    y,sr = librosa.load('demo.wav',sr=8000)
    y=y[3000:3000+sr]
    fig=plt.gcf()
    fig.set_size_inches(15,5)
    plt.plot(y)
    fig.savefig('wav.png')
    plt.close()
    print(run(y))

if __name__=='__main__':
    plotCNNFrameWork()
    
