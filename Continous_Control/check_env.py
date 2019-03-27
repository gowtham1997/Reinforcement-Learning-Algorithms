import gym
import pybullet_envs

ENV_ID = "MinitaurBulletEnv-v0"
RENDER = True

if __name__ == "__main__":
    spec = gym.envs.registry.spec(ENV_ID)
    spec._kwargs['render'] = True
    env = gym.make(ENV_ID)

    print('==================================================================')
    print()
    print(env.observation_space)
    print(env.action_space, env.action_space.sample())

    print(env.reset())
    print(env.action_space.sample())
    print()
    print('==================================================================')
    input('Press any key to exit\n')
    env.close()
