import pandas as pd
import matplotlib.pyplot as plt
import csv
import sys
import os

args = sys.argv[1:]
if len(args) != 4:
    print("Usage: python ", sys.argv[0], " <ppo csv> <dqn csv> <ddpg csv> <output_pngs_filename>")
    print("Description: plots the progress csv and stores it with same base filename in <output_pngs_dir> with .png extension")
    exit(1)

ppo_csv1 = args[0]
dqn_csv2 = args[1]
ddpg_csv3 = args[2]
output_png_file = args[3]
print("Progress CSVs: ", ppo_csv1, ", ", dqn_csv2, ", ", ddpg_csv3)
print("Output png file: ", output_png_file)

x = []
y = []
with open(ppo_csv1,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in plots:
        i += 1
        if i == 1:
            continue
        print(row[3], row[18])
        x.append(int(row[18]))
        y.append(float(row[3]))
        if int(row[18]) >= 120000:
          break
plt.plot(x,y, label='PPO', color='green')

x1=[]
y1=[]
with open(dqn_csv2,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in plots:
        i += 1
        if i == 1:
            continue
        x1.append(int(row[18]))
        y1.append(float(row[3]))
        if int(row[18]) >= 120000:
          break
plt.plot(x1,y1, label='DQN', color='steelblue')

x2=[]
y2=[]
with open(ddpg_csv3,'r') as csvfile:
    plots = csv.reader(csvfile, delimiter=',')
    i = 0
    for row in plots:
        i += 1
        if i == 1:
            continue
        x2.append(int(row[18]))
        y2.append(float(row[3]))
        if int(row[18]) >= 120000:
          break
plt.plot(x2,y2, label='DDPG', color='black')

plt.xlabel('Training episode')
plt.ylabel('Mean episode reward')
plt.legend()
plt.show()
plt.savefig(output_png_file)
