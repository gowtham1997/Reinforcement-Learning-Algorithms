import model
import wrappers
import gym
import time
import torch

ENV = 'PongNoFrameskip-v4'

if __name__ == "__main__":
    FPS = 60
    env = wrappers.make_env(ENV)
    env = gym.wrappers.Monitor(env, 'monitor')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = model.DQN(env.observation_space.shape, env.action_space.n)
    ck_path = 'checkpoints/' + ENV + '-best_normal.dat'
    print(ck_path)
    # load the saved model
    net.load_state_dict(torch.load(
        ck_path, map_location=lambda storage, loc: storage))
    print('model loaded')

    state = env.reset()
    while True:
        start_ts = time.time()
        state_v = torch.Tensor([state]).to(device)
        env.render()
        q_values = net(state_v)
        _, action_v = q_values.max(dim=1)
        action = action_v.item()
        new_state, _, is_done, _ = env.step(action)
        state = new_state
        delta = 1 / FPS - (time.time() - start_ts)
        if delta > 0:
            time.sleep(delta)
        if is_done:
            break

    env.env.close()
