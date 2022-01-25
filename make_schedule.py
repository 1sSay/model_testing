import numpy as np
import matplotlib.pyplot as plt
import glob


int_time = np.arange(9, 20, 2)
time = np.array([], dtype='str')
for i in int_time:
    hours = "0" + str(i) if len(str(i)) == 1 else str(i)   # Хочу, чтобы было не 9, а 09
    time = np.append(time, hours + ':00')

out = np.zeros((time.size, 3), float)
n = 0
for file in list(
        sorted(
            glob.glob(
                'schedule\\*.txt'), key=lambda x: int(x.split('\\')[-1][:-4]))):
    stat = open(file, 'r').read()
    out[n] = list(map(float, stat.split()))
    n += 1

fig, ax = plt.subplots()

ax.plot(time, out, lw=2, label=['Cars', 'Buses', 'Trucks'])
ax.legend(loc='upper left')
ax.set_ylabel('Number of vehicles in the one image')
ax.set_xlabel('Timeline')
ax.set_xlim(xmin=time[0], xmax=time[-1])
fig.tight_layout()

plt.show()
