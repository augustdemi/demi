import os
import numpy as np
kshot = "0/10shot/updatelr0.01.metalr0.005.numstep5"
x_range_test = [5,10,15,20,25]
#=================================================================================================
log_dir = "D:/연구/프로젝트/maml_result/" + kshot +"/train/"
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

#=================================================================================================
log_dir = "D:/연구/프로젝트/maml_result/" + kshot +"/test/"

iterations = os.listdir(log_dir)
iterations.sort(key=lambda iter:int(iter))
data_avg = {'outa':[], 'outb': []} # to have average data of 13 subjects



for out in outs:
    test_data = []
    for _ in iterations:
        test_data.append({"ICC" : [], "MSE" : [], "ACC" : [], "F1-b": []})

    for i in range(len(iterations)):
        files = []
        iter = iterations[i]
        for file_name in os.listdir(log_dir + iter + "/"):
            if (file_name.startswith(out)):
                files.append((int(file_name.split(".")[0].split("_")[1]), file_name))
        print(len(files))
        files.sort(key=lambda f:f[0])
        for f in files:
            print(log_dir + iter + "/"+f[1])
            file = open(log_dir + iter + "/"+f[1])
            for line in file:
                if(line.startswith("ICC")):
                    fig = line.split(" ")
                    test_data[i]['ICC'].append(float(fig[len(fig)-1]))
                elif(line.startswith("RMSE")):
                    fig = line.split(" ")
                    test_data[i]['MSE'].append(float(fig[len(fig)-1]))
                elif(line.startswith("ACC")):
                    fig = line.split(" ")
                    test_data[i]['ACC'].append(float(fig[len(fig)-1]))
                elif(line.startswith("F1-b")):
                    fig = line.split(" ")
                    test_data[i]['F1-b'].append(float(fig[len(fig)-1]))
        # to have average data of 13 subjects
        data_avg_one = {}
        for key in test_data[i]:
            data_avg_one[key] = np.average(test_data[i][key])
        data_avg[out].append(data_avg_one)
#=================================================================================================

import matplotlib.pyplot as plt
plt.figure()
x_range = list(range(1, N_files+1))
start = N_files/len(iterations)

plt.suptitle("train & sbjt-avged test: " + kshot, fontsize=16)

plt.subplot(221)
print(data[0]['ICC'])
print(data[1]['ICC'])
plt.plot(x_range, data[0]['ICC'], "r*", alpha=0.7)
plt.plot(x_range, data[0]['ICC'], color ="red", alpha=0.3, label = "outa")
plt.plot(x_range, data[1]['ICC'], "b*", alpha=0.7)
plt.plot(x_range, data[1]['ICC'], color = "blue", alpha=0.3, label = "outb")
test_outa_avg = [data_avg['outa'][i]['ICC'] for i in range(len(iterations))]
test_outb_avg = [data_avg['outb'][i]['ICC'] for i in range(len(iterations))]
plt.plot(x_range_test, test_outa_avg, "ro", alpha=0.7)
plt.plot(x_range_test, test_outa_avg, color ="red", alpha=0.3)
plt.plot(x_range_test, test_outb_avg, "bo", alpha=0.7)
plt.plot(x_range_test, test_outb_avg, color = "blue", alpha=0.3)

plt.xticks(range(1, N_files+1, 1))
plt.legend(loc=0)
plt.xlabel('num of meta train iter(*100)')
plt.ylabel('ICC')
plt.title('ICC')


plt.subplot(222)
plt.plot(x_range, data[0]['MSE'], "r*", alpha=0.7)
plt.plot(x_range, data[0]['MSE'], color ="red", alpha=0.3, label = "outa")
plt.plot(x_range, data[1]['MSE'], "b*", alpha=0.7)
plt.plot(x_range, data[1]['MSE'], color = "blue", alpha=0.3, label = "outb")
test_outa_avg = [data_avg['outa'][i]['MSE'] for i in range(len(iterations))]
test_outb_avg = [data_avg['outb'][i]['MSE'] for i in range(len(iterations))]
plt.plot(x_range_test, test_outa_avg, "ro", alpha=0.7)
plt.plot(x_range_test, test_outa_avg, color ="red", alpha=0.3)
plt.plot(x_range_test, test_outb_avg, "bo", alpha=0.7)
plt.plot(x_range_test, test_outb_avg, color = "blue", alpha=0.3)

plt.xticks(range(1, N_files+1, 1))
plt.legend(loc=1)
plt.xlabel('num of meta train iter(*100)')
plt.ylabel('RMSE')
plt.title('RMSE')


plt.subplot(223)
plt.plot(x_range, data[0]['ACC'], "r*", alpha=0.7)
plt.plot(x_range, data[0]['ACC'], color ="red", alpha=0.3, label = "outa")
plt.plot(x_range, data[1]['ACC'], "b*", alpha=0.7)
plt.plot(x_range, data[1]['ACC'], color = "blue", alpha=0.3, label = "outb")
test_outa_avg = [data_avg['outa'][i]['ACC'] for i in range(len(iterations))]
test_outb_avg = [data_avg['outb'][i]['ACC'] for i in range(len(iterations))]
plt.plot(x_range_test, test_outa_avg, "ro", alpha=0.7)
plt.plot(x_range_test, test_outa_avg, color ="red", alpha=0.3)
plt.plot(x_range_test, test_outb_avg, "bo", alpha=0.7)
plt.plot(x_range_test, test_outb_avg, color = "blue", alpha=0.3)

plt.xticks(range(1, N_files+1, 1))
plt.legend(loc=0)
plt.xlabel('num of meta train iter(*100)')
plt.ylabel('ACC')
plt.title('ACC')


plt.subplot(224)
plt.plot(x_range, data[0]['F1-b'], "r*", alpha=0.7)
plt.plot(x_range, data[0]['F1-b'], color ="red", alpha=0.3, label = "outa")
plt.plot(x_range, data[1]['F1-b'], "b*", alpha=0.7)
plt.plot(x_range, data[1]['F1-b'], color = "blue", alpha=0.3, label = "outb")
test_outa_avg = [data_avg['outa'][i]['F1-b'] for i in range(len(iterations))]
test_outb_avg = [data_avg['outb'][i]['F1-b'] for i in range(len(iterations))]
plt.plot(x_range_test, test_outa_avg, "ro", alpha=0.7)
plt.plot(x_range_test, test_outa_avg, color ="red", alpha=0.3)
plt.plot(x_range_test, test_outb_avg, "bo", alpha=0.7)
plt.plot(x_range_test, test_outb_avg, color = "blue", alpha=0.3)
plt.xticks(range(1, N_files+1, 1))
plt.legend(loc=0)
plt.xlabel('num of meta train iter(*100)')
plt.ylabel('F1-b')
plt.title('F1-b')


plt.show()