""" Find the point at which training episodes converged.
"""

from collections.abc import Sequence

from absl import app


def main(argv: Sequence[str]) -> None:
  if len(argv) > 2:
    raise app.UsageError('Too many command-line arguments.')
  if len(argv) != 2:
    raise app.UsageError('Usage: ', argv[0], " <progress_csv_filename>")

  filename = argv[1]

  f=open(filename,"r")
  lines=f.readlines()
  f.close()

  best_iter = 0
  best_episodes_toal = 0
  best_mean = -1000000.0
  i = 0
  for x in lines:
    cols = x.split(',')

    if i==0:
      print(cols[19], " ", cols[18], " ", cols[3])
      i += 1
      continue

    training_iteration = int(cols[19])
    episodes_total = int(cols[18])
    episode_reward_mean = float(cols[3])
    if best_mean != 0 and episode_reward_mean > best_mean:
      best_mean = episode_reward_mean
      best_iter = training_iteration
      best_episodes_toal = episodes_total
    i += 1

  print("Best mean = ", best_mean, " at iteration = ", best_iter, " episodes_total = ", episodes_total)

if __name__ == '__main__':
  app.run(main)
