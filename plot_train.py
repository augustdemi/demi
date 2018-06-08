import os
import numpy as np
# kshot = "5shot_uplr0.005"
kshot = "10shot/updatelr0.005.metalr0.005.numstep5"

#=================================================================================================
log_dir = "D:/연구/프로젝트/maml_result/1/" + kshot +"/train/"
N_files = 0

outs = ['outa', 'outb']
data = []
for _ in range(len(outs)):
    data.append({"ICC" : [], "MSE" : [], "ACC" : [], "F1-b": []})

for i in range(len(outs)):
    files = []
    out = outs[i]
    for file_name in os.listdir(log_dir):
        if (file_name.startswith(out)):
            files.append((int(file_name.split(".")[0].split("_")[1]), file_name))
    print(len(files))
    files.sort(key = lambda f:f[0])
    N_files = len(files)
    for f in files:
        file = open(log_dir + f[1])
        for line in file:
            if(line.startswith("ICC")):
                fig = line.split(" ")
                data[i]['ICC'].append(float(fig[len(fig)-1]))
            elif(line.startswith("RMSE")):
                fig = line.split(" ")
                data[i]['MSE'].append(float(fig[len(fig)-1]))
            elif(line.startswith("ACC")):
                fig = line.split(" ")
                data[i]['ACC'].append(float(fig[len(fig)-1]))
            elif(line.startswith("F1-b")):
                fig = line.split(" ")
                data[i]['F1-b'].append(float(fig[len(fig)-1]))


import matplotlib.pyplot as plt
plt.figure()
x_range = list(range(1, N_files+1))
plt.suptitle(kshot + " - train", fontsize=16)

plt.subplot(221)
print(data[0]['ICC'])
print(data[1]['ICC'])
plt.plot(x_range, data[0]['ICC'], "o")
plt.plot(x_range, data[0]['ICC'],label = outs[0])
plt.plot(x_range, data[1]['ICC'], "r*")
plt.plot(x_range, data[1]['ICC'], label = outs[1])
plt.xticks(range(1, N_files+1, 1))
plt.legend(loc=0)
plt.xlabel('num of iterations(*100)')
plt.ylabel('ICC')
plt.title('ICC')


plt.subplot(222)
y = data[0]['MSE']
plt.plot(x_range, data[0]['MSE'], "o")
plt.plot(x_range, data[0]['MSE'],label = outs[0])
plt.plot(x_range, data[1]['MSE'], "r*")
plt.plot(x_range, data[1]['MSE'], label = outs[1])
plt.xticks(range(1, N_files+1, 1))
plt.legend(loc=1)
plt.xlabel('num of iterations(*100)')
plt.ylabel('RMSE')
plt.title('RMSE')


plt.subplot(223)
plt.plot(x_range, data[0]['ACC'], "o")
plt.plot(x_range, data[0]['ACC'],label = outs[0])
plt.plot(x_range, data[1]['ACC'], "r*")
plt.plot(x_range, data[1]['ACC'], label = outs[1])
plt.xticks(range(1, N_files+1, 1))
plt.legend(loc=0)
plt.xlabel('num of iterations(*100)')
plt.ylabel('ACC')
plt.title('ACC')


plt.subplot(224)
plt.plot(x_range, data[0]['F1-b'], "o")
plt.plot(x_range, data[0]['F1-b'],label = outs[0])
plt.plot(x_range, data[1]['F1-b'], "r*")
plt.plot(x_range, data[1]['F1-b'], label = outs[1])
plt.xticks(range(1, N_files+1, 1))
#plt.yticks(range(0, 1, 10))
plt.legend(loc=0)
plt.xlabel('num of iterations(*100)')
plt.ylabel('F1-b')
plt.title('F1-b')


plt.show()