import os
import pickle
import librosa

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

pickle_file = "./user_data/recordings.pickle"


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

