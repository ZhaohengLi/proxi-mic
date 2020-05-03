# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import librosa.display

# CnnNet5


class CnnNet8(nn.Module):
    def __init__(self, in_channels=20, out_channels=2):
        super(CnnNet8, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=50, kernel_size=3)
        self.batchnorm1 = nn.BatchNorm1d(num_features=50)
        self.pool1 = nn.MaxPool1d(kernel_size=3)
        self.conv2 = nn.Conv1d(
            in_channels=50, out_channels=100, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm1d(num_features=100)
        self.pool2 = lambda x: torch.max(x, 2)[0]
        self.fc1 = nn.Linear(100, 20)
        self.fc2 = nn.Linear(20, out_channels)

    def forward(self, x):
        x = self.pool1(F.relu(self.batchnorm1(self.conv1(x))))
        x = self.pool2(F.relu(self.batchnorm2(self.conv2(x))))
        x = x.view(-1, 100)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model_path = './speech-5.model'
print('load CNN.')
net = CnnNet8(in_channels=20, out_channels=2)
net.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# assert(torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)
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
        return weight
    S = librosa.amplitude_to_db(
        np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length)))
    S = np.dot(FilterBank(sr, n_fft, n_feature=20), S)
    # Create a Filterbank matrix to combine FFT bins
    result = torch.from_numpy(S)
    return result


def run(y: np.ndarray,threshold=2.0) -> int:
    assert(len(y) == 8000)
    F = FilterBankFeature(y, sr=8000)
    F = torch.unsqueeze(F, 0)
    F = F.to(device)
    with torch.no_grad():
        outputs = net(F)  #[[activate_p,reject_p]]
    r=outputs.cpu()[0].numpy()
    if r[0]-r[1]>threshold:
        return 0 #activate
    else:
        return 1

def run_p(y: np.ndarray):
    assert(len(y) == 8000)
    F = FilterBankFeature(y, sr=8000)
    F = torch.unsqueeze(F, 0)
    F = F.to(device)
    with torch.no_grad():
        outputs = net(F)
    return outputs.cpu()[0].numpy()

def test():
    y=np.random.random(8000)*0.2
    print(run(y),'  (0:activate 1:silent)')


if __name__ == '__main__':
    test()
