import csv
import config
import glob
import os

def read_csv(filename):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        f.close()
    return data

def combine_csv(batch_num):
    os.chdir(config.results_folder)
    filelist = glob.glob('*.csv')
    data = read_csv(filelist[0])
    #print(filelist[0])
    m = len(data[0])
    #print(m)
    #print(data[0])
    n = len(data[0][0].split(","))
    combined = []
    for i in range(9):
        row = []
        for j in range(m):
            ar = []
            for l in range(n):
                ar.append(0)
            row.append(ar)
        combined.append(row)


    for filename in filelist:
        with open(filename, newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
            f.close()
            for i in range(9):
                for j in range (m):
                    vec = str(data[i][j]).replace("[","").replace("]","").split(",")
                    for l in range(n):
                        combined[i][j][l] += int(vec[l])
    with open('combined.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(combined)
        f.close()