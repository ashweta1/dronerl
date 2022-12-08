import pandas as pd
import matplotlib.pyplot as plt
import csv
import sys
import os

x = []
y = []

args = sys.argv[1:]
if len(args) != 2:
    print("Usage: python ", sys.argv[0], " <progress_csv_filename> <output_pngs_dir>")
    print("Description: plots the progress csv and stores it with same base filename in <output_pngs_dir> with .png extension")
    exit(1)

progress_csv = args[0]
output_pngs_dir = args[1]
print("Progress CSV: ", progress_csv)
print("Output pngs dir: ", output_pngs_dir)

base_filename = os.path.splitext(os.path.basename(progress_csv))[0]
output_png = os.path.join(output_pngs_dir, base_filename + ".png")
print("Output png file name: ", output_png)

columns = ["training_iteration", "episode_reward_mean", "episodes_total"]
df = pd.read_csv(progress_csv, usecols=columns)
print("Contents in csv file: ", df)
print("Contents in csv file shape: ", df.shape)

plt.rcParams["figure.autolayout"] = True

col1 = 'steelblue'
ax = df.plot()
ax.set_xlabel("Training episode")
ax.set_ylabel("Mean episode reward")
ax.plot(df.episodes_total, df.episode_reward_mean, color=col1)
plt.show()
plt.savefig(output_png)
plt.close()
