# ppo_pendulum

## Download this repository and move the file

ppo_pendulum-dones[t].ipynb

to upper directory.


## Your work directory should look as this
```
$ ls
baselines
ppo_pendulum
ppo_pendulum-dones[t].ipynb
```

## Run and reference with

```
ppo_pendulum-dones[t].ipynb
```

## Version

### nsteps & ly3

1. Increase nsteps from 512 to 2048

2. Add one layer in value networks and policy networks.

```
-        nsteps = 512
+        nsteps = 2048

+            ly3 = tf.layers.dense(ly2, act_dim, activation=None, name='ly3')
+            action_nodes = ly3

+        ly3 = tf.layers.dense(ly2, 1, activation=None, name='ly3')
+        ly3 = tf.reshape(ly3, [-1])
+        return ly3
```