import os

# log_dir = "D:/연구/프로젝트/DISFA/rrr/4/"
log_dir = "D:/연구/프로젝트/log/disfa_soft/"

ICC = []
MSE = []
ACC = []

te = {"ICC" : [], "MSE" : [], "ACC" : [], "F1-b": []}
tr = {"ICC" : [], "MSE" : [], "ACC" : [], "F1-b": []}
data = [tr, te]
#
# for file_name in os.listdir(log_dir):
#     t= file_name.split(".")[0]
#     sp = t.split("_")
#     os.rename(log_dir+file_name, log_dir+"_".join(sp[0:1]) +  sp[2].zfill(4) + '.txt')
#
#
set = ['TR', 'TE']
for i in range(0,len(set)):
    files = []
    data_set = set[i]
    for file_name in os.listdir(log_dir):
        if (file_name.startswith(data_set)):
            s = file_name.split(".")[0]
            # if (int(s.split(data_set)[1].split("_")[2]) % 5 == 0 ):
            files.append(file_name)
    print(len(files))
    for file_name in files:
        print(file_name)
        f = open(log_dir + file_name)
        for line in f:
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


iter = 78
import matplotlib.pyplot as plt
plt.figure(figsize=(12,12))
x_range = list(range(0, iter))


plt.subplot(221)
print(data[0]['ICC'])
print(data[1]['ICC'])
plt.plot(x_range, data[0]['ICC'], "o")
plt.plot(x_range, data[0]['ICC'],label = "train")
plt.plot(x_range, data[1]['ICC'], "r*")
plt.plot(x_range, data[1]['ICC'], label = "test")



plt.legend(loc=0)
plt.xlabel('Number of training iteration')
plt.ylabel('ICC')
plt.title('ICC')


plt.subplot(222)
#plt.axes([0, 80, 0, 1])

y = data[0]['MSE']
plt.plot(x_range, data[0]['MSE'], "o")
plt.plot(x_range, data[0]['MSE'],label = "train")
plt.plot(x_range, data[1]['MSE'], "r*")
plt.plot(x_range, data[1]['MSE'], label = "test")
#plt.axis([0, 80, 0.1, 1.0])
plt.xticks(range(0, iter, 10))
#plt.yticks(range(0, 1, 10))
plt.legend(loc=1)
plt.xlabel('Number of training iteration')
plt.ylabel('RMSE')
plt.title('RMSE')


plt.subplot(223)
plt.plot(x_range, data[0]['ACC'], "o")
plt.plot(x_range, data[0]['ACC'],label = "train")
plt.plot(x_range, data[1]['ACC'], "r*")
plt.plot(x_range, data[1]['ACC'], label = "test")
plt.xticks(range(0, iter, 10))
#plt.yticks(range(0, 1, 10))
plt.legend(loc=0)
plt.xlabel('Number of training iteration')
plt.ylabel('ACC')
plt.title('ACC')


plt.subplot(224)
plt.plot(x_range, data[0]['F1-b'], "o")
plt.plot(x_range, data[0]['F1-b'],label = "train")
plt.plot(x_range, data[1]['F1-b'], "r*")
plt.plot(x_range, data[1]['F1-b'], label = "test")
plt.xticks(range(0, iter, 10))
#plt.yticks(range(0, 1, 10))
plt.legend(loc=0)
plt.xlabel('Number of training iteration')
plt.ylabel('F1-b')
plt.title('F1-b')


#
#
# # latent space
# import pickle
# # log_dir = "D:/연구/프로젝트/log/disfa/z_val/latent.pkl"
# log_dir = "D:/연구/프로젝트/log/mnist/mnist_log/z_val/latent.pkl"
# data=pickle.load(open(log_dir, "rb" ), encoding='latin1')
#
#
# Z=data['z']
# Y = data['y']
# # S = data['s']
# # S= np.array(S[0])
# # S =S.reshape(S.shape[0] * S.shape[1],)
# print(len(Y))
# from sklearn.manifold import TSNE
# onehot = Y
#
# import numpy as np
# label = []
# for y in onehot:
#     label.append(np.argmax(y))
#
# model = TSNE(n_components=2, random_state=0)
# reduced_z = model.fit_transform(Z)
# x = reduced_z[:,0]
# y = reduced_z[:,1]
#
#
# plt.subplot(224)
# plt.title('Latent Space Z0')
# plt.scatter(x, y, c = label,cmap="gist_rainbow")


plt.show()