import os

lable_dir = '/home/ml1323/project/robert_data/DISFA/label/'
# lable_dir = 'D:/연구/프로젝트/SN001/label/'
subjects = os.listdir(lable_dir)

num_pos_all = {}
f_num_pos_all = open('/home/ml1323/project/robert_data/DISFA/disfa_label_summary.csv', "w")

for subject in subjects:
    files = []
    subject = subject.split('_')[0]
    for file_name in os.listdir(lable_dir + subject):
        files.append((int(file_name.split('.')[0].split('_')[1].split('au')[1]), file_name))
    files.sort(key=lambda f: f[0])
    print(files)
    num_pos = [0] * 12
    for i in range(len(files)):  # for each au
        f = open(lable_dir + subject + '/' + files[i][1])  # 각 subject별 한 au 파일
        while True:
            line = f.readline().split(',')
            if len(line) == 1: break
            intensity = int(line[1])
            if intensity > 0: num_pos[i] += 1
    num_pos_all.update({subject: num_pos})
    f_num_pos_all.write(subject + ',' + ','.join(str(x) for x in num_pos) + '\n')

print(num_pos_all)
