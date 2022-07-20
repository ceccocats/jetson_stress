import matplotlib.pyplot as plt

def timestamp2seconds(vec):
    startTs = vec[0][0]
    for i in range(len(vec)):
        vec[i][0] = float(vec[i][0] - startTs) / 1e9
    return vec

def avg(vec):
    return sum(vec)/len(vec)

cpu =  timestamp2seconds([ [int(j[0]), float(j[1])] for j in [ i.strip().split(" ") for i in open("cpu_times.txt").readlines()] ])
gpu =  timestamp2seconds([ [int(j[0]), float(j[1])] for j in [ i.strip().split(" ") for i in open("gpu_times.txt").readlines()] ])
data = timestamp2seconds([ [int(j[0]), j[10], j[14], j[15], j[16], j[17], j[18], j[19], j[20], j[21]] for j in [ i.strip().split(" ") for i in open("temps.txt").readlines()] ])

# cpu percetage fix
for j in range(len(data)):
    data[j][1] = str(int(avg([ float(i.split("%")[0]) for  i in data[j][1].strip("[]").split(",") ]))) +"%"

cpu_load = list(zip(list(zip(*data))[0], [ float(i.strip("%")) for i in list(zip(*data))[1] ] ))
gpu_load = list(zip(list(zip(*data))[0], [ float(i.strip("%")) for i in list(zip(*data))[2] ] ))

temps = {}
alltemps = list(zip(*data))[-7:]
for i in alltemps:
    name = i[0].split("@")[0]
    temps[name] = [ float(j.split("@")[1].strip("C")) for j in i ]


fig, axs = plt.subplots(2, 2)
axs[0, 0].boxplot(list(zip(*cpu))[1])
axs[0, 0].set_title('cpu time (s)')

axs[0, 1].boxplot(list(zip(*gpu))[1])
axs[0, 1].set_title('gpu time (s)')

axs[1, 0].plot(list(zip(*cpu_load))[0], list(zip(*cpu_load))[1], label='cpu')
axs[1, 0].plot(list(zip(*gpu_load))[0], list(zip(*gpu_load))[1], label='gpu')
axs[1, 0].legend()
axs[1, 0].set_title('system load (%)')


show = [ "GPU", "CPU", "iwlwifi"] #, "thermal" ]
for t in temps.keys():
    if(t in show):
        axs[1, 1].plot(list(zip(*cpu_load))[0], temps[t], label=t)
axs[1, 1].legend()
axs[1, 1].set_title('temperatures (celsius)')

plt.show()

