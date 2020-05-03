import os
import pickle
import librosa

folders = ["./recordings/2cm-0/",
           "./recordings/2cm-cover/",
           "./recordings/2cm-cover-new/",
           "./recordings/2cm-new/",
           "./recordings/5cm-0/",
           "./recordings/5cm-30/",
           "./recordings/5cm-60/",
           "./recordings/5cm-90/",
           "./recordings/10cm-0/",
           "./recordings/10cm-small/",
           "./recordings/20cm-0/",
           "./recordings/30cm-0/",
           "./recordings/50cm-0/",
           "./recordings/attack/"]

pickle_file = "./recordings/recordings.pickle"


def to_pickle():
    data = {}

    for folder in folders:
        print(folder)
        recordings = []
        for file_name in find_all_m4a_file(folder):
            y, sr = librosa.load(file_name, sr=None, mono=False)
            assert (sr == 48000)
            y = y[0].copy() if y[0].max() > y[1].max() else y[1].copy()
            for i in range(6):
                section = y[i::6]
                recordings.append(section)
        data[folder] = recordings

    with open(pickle_file, "wb") as file:
        pickle.dump(data, file)


def from_pickle():
    with open(pickle_file, "rb") as file:
        data = pickle.load(file)
        return data


def find_all_m4a_file(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            if f.endswith('.m4a'):
                fullname = os.path.join(root, f)
                yield fullname


if __name__ == '__main__':
    from_pickle()
    pass

