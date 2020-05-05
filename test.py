from model import run
from model import model_path
import math
from convert import from_pickle

log_path = "./analysis/activate_rate.txt"


def test_frame():
    with open(log_path, "a") as file:
        file.write("frame activate rate is as follows.\n")

    data = from_pickle()

    for folder in data.keys():
        activate_sum = 0
        frame_sum = 0

        for recording in data[folder]:
            sr = 8000
            th = 0.05
            r = 0.1
            s = max(0, (recording < th).argmin() - int(r * sr))
            t = min(-1, -((recording < th)[::-1].argmin()) + int(r * sr))
            recording = recording[s:t]

            if len(recording) < 8000:
                continue

            section_num = math.ceil(len(recording) / 8000)
            for i in range(section_num):
                frame_sum += 1
                if i == section_num-1:
                    result = run(recording[-8000:])
                else:
                    result = run(recording[i*8000:(i+1)*8000])
                if result == 0:
                    activate_sum += 1

        with open(log_path, "a") as file:
            file.write(folder+": "+str(activate_sum)+"/"+str(frame_sum)+" = "+str(activate_sum / frame_sum) + "\n")


def test_recording():
    with open(log_path, "a") as file:
        file.write("recording activate rate is as follows.\n")

    data = from_pickle()

    for folder in data.keys():
        recording_sum = 0
        activate_sum = 0
        for recording in data[folder]:
            if len(recording) < 8000:
                continue

            recording_sum += 1
            section_num = math.ceil(len(recording) / 8000)
            for i in range(section_num):
                if i == section_num-1:
                    result = run(recording[-8000:])
                else:
                    result = run(recording[i*8000:(i+1)*8000])
                if result == 0:
                    activate_sum += 1
                    break

        with open(log_path, "a") as file:
            file.write(folder + ": "+str(activate_sum)+"/"+str(recording_sum)+" = "+str(activate_sum / recording_sum) + "\n")


if __name__ == '__main__':
    with open(log_path, "a") as file:
        file.write("\n"+model_path+"\n")
    test_frame()
    test_recording()
