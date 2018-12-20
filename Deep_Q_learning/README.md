# Double Deep Q Learning For Atari
This repo contains code in pytorch(0.4) for training an agent to play atari games using DDQN. This is an implementation of [Deepmind's paper](https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).

## Folder Structure

```
--- monitor -> tracks metadata and store videos of the agent (not tracked with git)
--- __pycache__ (not tracked)
--- checkpoints -> stores model checkpoints(not tracked)
--- runs -> logs for Tensorboard (not tracked)
--- wrappers.py -> wrappers to modify the environment
 model.py -> contains the model
 dqn_pong.py -> for training the model
 dqn_play.py -> inference
 requirements.txt -> dependencies
 .gitignore -> files to be ignored by git
```

## Getting Started

Clone the repo and run `dqn_pong.py` with right params to train the model.

Run `tensorboard --logdir runs --host localhost` to monitor training and tune your hyperparmeters

Run `dqn_play.py` with the model_path for the inference.

### Prerequisites

Run `pip install requirements.txt` to get all dependencies. They include:
```
torch==0.4.1
numpy==1.15.4
opencv_python==3.4.0.14
gym==0.10.9
tensorboardX==1.5
```

### Summary

The model takes screen frames(4 frames) as input and returns Q_values for every possible action.

Q value represents the value associated with a state and an action. It is defined as the immediate reward obtained after performing an action *a* in a state *s* and **acting optimally** thereafter.

*`Q(s, a) = r + gamma * Q ^* (s', a')`*

where 
    
    r is the immediate reward,

    gamma is discount factor(how much you care about the futuer reward),

    s, a -> current state and action

    s', a' -> next state and optimal action at state s'

We estimate `Q'(s, a)` with the neural net and make it closer to `Q(s, a)`(see formula above).

The exploration-explotation tradeoff is handled by choosing between a policy and acting randomly based on value of *epsilon*. Epsilon is made to 1.0(100% random actions) at start and linearly decreased to 0.02.

The model training takes about *2 hours* on a single V100 GPU trained at speeds of *~90 frames/second*.




