# ppo_pendulum

## Download this repository and move the files

- ppo_pendulum-nsteps & ly3.ipynb

- ppo_pendulum-dones[t].ipynb

to upper directory.


## Your work directory should look as this
```
$ ls
baselines
ppo_pendulum
ppo_pendulum-dones[t].ipynb
ppo_pendulum-nsteps & ly3.ipynb
```

## Run and reference
Reset to related version with git, and details please refer to the .ipynb file you use.

## Version

### nsteps & ly3

1. Increase nsteps from 512 to 2048

2. Add one new layer in value networks and policy networks.

```
-        nsteps = 512
+        nsteps = 2048

+            ly3 = tf.layers.dense(ly2, act_dim, activation=None, name='ly3')
+            action_nodes = ly3

+        ly3 = tf.layers.dense(ly2, 1, activation=None, name='ly3')
+        ly3 = tf.reshape(ly3, [-1])
+        return ly3
```