# risk-averse-RL

Risk-averse reinforcement learning algorithms for drone indoor navigation and obstacle avoidance.

- [ ] move replay_memory to utils


## PID killer

```
lsof -i :1511 # find which process is using the port
kill PID # kill the process
```

## experiment 171

works good, reward function good. training loss good.
branch reflect.

**trained in no obstacle environment but has the power to generalize to obstacle environments/different goal positions although the path is suboptimal! but it can avoid unseen obstacles!
run test:

```shell
python3.6 test.py --dir=171 --distortion=neutral --render_mode=trajectory --num_speeds=3 --seed=1
```