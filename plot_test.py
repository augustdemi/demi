import os
import matplotlib.pyplot as plt
import numpy as np
# kshot = "0shot/robert"
kshot = "5shot/updatelr0.01.metalr0.005.numstep5"
log_dir = "D:/연구/프로젝트/maml_result/"


paths = [log_dir+"1/0shot/", log_dir + "1/10shot/updatelr0.005.metalr0.005.numstep5/test/400/"]
# paths = [log_dir+"1/0shot/", log_dir + "0/10shot/updatelr0.01.metalr0.005.numstep5/test/500/"]
iterations = ['0shot','10shot']

data = []
for _ in range(len(paths)):
    data.append({"ICC" : [], "MSE" : [], "ACC" : [], "F1-b": []})

for i in range(len(paths)):
    files = []
    for file_name in os.listdir(paths[i]):
        if (file_name.startswith("outa")):
            files.append((int(file_name.split(".")[0].split("_")[1]), file_name))
    print(len(files))

    files.sort(key=lambda f:f[0])
    for f in files:
        print(paths[i] + f[1])
        file = open(paths[i] + f[1])
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
    print("===================")


subjects = 13
opacity = 0.6
bar_width = 0.3
plt.figure(figsize = (10,7))
x_range = np.arange(0, subjects)
plt.suptitle("compare test results for each subject", fontsize=16)


plt.subplot(221)
plt.bar(x_range - bar_width/2, data[0]['ICC'], bar_width, alpha=opacity, color = "blue", label = iterations[0])
plt.bar(x_range + bar_width/2, data[1]['ICC'], bar_width, alpha=opacity, color= "red", label = iterations[1])
# plt.plot(x_range, data[0]['ICC'],label = iterations[0])
# plt.plot(x_range, data[1]['ICC'], label = iterations[1])
plt.xticks(range(0, subjects, 1))
plt.legend(loc=0)
plt.xlabel('Subject')
plt.ylabel('ICC')
plt.title('ICC')
# plt.tight_layout()


plt.subplot(222)
y = data[0]['MSE']
plt.bar(x_range - bar_width/2, data[0]['MSE'], bar_width, alpha=opacity, color = "blue", label = iterations[0])
plt.bar(x_range + bar_width/2, data[1]['MSE'], bar_width, alpha=opacity, color= "red", label = iterations[1])
# plt.plot(x_range, data[0]['MSE'], "o")
# plt.plot(x_range, data[0]['MSE'],label = iterations[0])
# plt.plot(x_range, data[1]['MSE'], "r*")
# plt.plot(x_range, data[1]['MSE'], label = iterations[1])
plt.xticks(range(0, subjects, 1))
plt.legend(loc=1)
plt.xlabel('Subject')
plt.ylabel('RMSE')
plt.title('RMSE')


plt.subplot(223)
plt.bar(x_range - bar_width/2, data[0]['ACC'], bar_width, alpha=opacity, color = "blue", label = iterations[0])
plt.bar(x_range + bar_width/2, data[1]['ACC'], bar_width, alpha=opacity, color= "red", label = iterations[1])
# plt.plot(x_range, data[0]['ACC'], "o")
# plt.plot(x_range, data[0]['ACC'],label = iterations[0])
# plt.plot(x_range, data[1]['ACC'], "r*")
# plt.plot(x_range, data[1]['ACC'], label = iterations[1])
plt.xticks(range(0, subjects, 1))
plt.legend(loc=0)
plt.xlabel('Subject')
plt.ylabel('ACC')
plt.title('ACC')


plt.subplot(224)
plt.bar(x_range - bar_width/2, data[0]['F1-b'], bar_width, alpha=opacity, color = "blue", label = iterations[0])
plt.bar(x_range + bar_width/2, data[1]['F1-b'], bar_width, alpha=opacity, color= "red", label = iterations[1])
# plt.plot(x_range, data[0]['F1-b'], "o")
# plt.plot(x_range, data[0]['F1-b'],label = iterations[0])
# plt.plot(x_range, data[1]['F1-b'], "r*")
# plt.plot(x_range, data[1]['F1-b'], label = iterations[1])
plt.xticks(range(0, subjects, 1))
#plt.yticks(range(0, 1, 10))
plt.legend(loc=0)
plt.xlabel('Subject')
plt.ylabel('F1-b')
plt.title('F1-b')


plt.show()