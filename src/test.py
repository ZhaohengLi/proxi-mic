import os
from model import run
from model import model_path
import librosa
import math
from convert import from_pickle

folders = ["./user_data/2cm-0/",
           "./user_data/2cm-cover/",
           "./user_data/2cm-cover-new/",
           "./user_data/2cm-new/",
           "./user_data/5cm-0/",
           "./user_data/5cm-30/",
           "./user_data/5cm-60/",
           "./user_data/5cm-90/",
           "./user_data/10cm-0/",
           "./user_data/10cm-small/",
           "./user_data/20cm-0/",
           "./user_data/30cm-0/",
           "./user_data/50cm-0/",
           "./user_data/attack/"]

log_path = "./output/activate_rate.txt"


def test_frame():
    with open(log_path, "a") as file:
        file.write("frame activate rate is as follows.\n")

    for folder in folders:
        activate_sum = 0
        frame_sum = 0

        for file_name in find_all_m4a_file(folder):
            y, sr = librosa.load(file_name, sr=None, mono=False)
            y = y[0].copy() if y[0].max() > y[1].max() else y[1].copy()
            th = 0.0001
            r = 0.1
            s = max(0, (y < th).argmin()-int(r*sr))
            t = min(-1, -((y < th)[::-1].argmin())+int(r*sr))
            y = y[s:t]

            if len(y) < 8000:
                print(file_name)
                continue

            section_num = math.ceil(len(y) / 8000)
            for i in range(section_num):
                frame_sum += 1
                if i == section_num-1:
                    result = run(y[-8000:])
                else:
                    result = run(y[i*8000:(i+1)*8000])
                if result == 0:
                    activate_sum += 1

        with open(log_path, "a") as file:
            file.write(folder + ": " + str(activate_sum / frame_sum) + "\n")


def test_recording():
    with open(log_path, "a") as file:
        file.write("recording activate rate is as follows.\n")

    for folder in folders:
        recording_sum = 0
        activate_sum = 0
        for file_name in find_all_m4a_file(folder):
            recording_sum += 1
            y, sr = librosa.load(file_name, sr=8000, mono=False)
            y = y[0].copy() if y[0].max() > y[1].max() else y[1].copy()

            if len(y) < 8000:
                print(file_name)
                continue

            section_num = math.ceil(len(y) / 8000)
            for i in range(section_num):
                if i == section_num-1:
                    result = run(y[-8000:])
                else:
                    result = run(y[i*8000:(i+1)*8000])
                if result == 0:
                    activate_sum += 1
                    break

        with open(log_path, "a") as file:
            file.write(folder + ": " + str(activate_sum / recording_sum) + "\n")


def find_all_m4a_file(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.m4a'):
                fullname = os.path.join(root, f)
                yield fullname


if __name__ == '__main__':

    with open(log_path, "a") as file:
        file.write("\n"+model_path+"\n")
    test_frame()
    test_recording()
