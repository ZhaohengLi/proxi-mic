import matplotlib.pyplot as plt
from convert import from_pickle
import numpy as np
data = from_pickle()

recordings = data.get("./recordings/5cm-pen-re/")
recording = recordings[600]
window = 1000
bar_l = 1
bar_h = 1
section = 0
for i in range(int(len(recording)/window)):
    sum = np.sum(recording[i*window:(i+1)*window]**2)
    if sum > bar_h:
        break
    if sum <= bar_l:
        section = i

recording = recording[(section+1)*window:]

sr = 8000
th = 0.05
r = 0.1
s = max(0, (recording < th).argmin() - int(r * sr))
t = min(-1, -((recording < th)[::-1].argmin()) + int(r * sr))
recording = recording[s:t]
plt.plot(recording)
plt.show()