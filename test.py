import gym
env = gym.make('gym_QTank:QTank-v0')

print(env.info)
for i_episode in range(1):
    observation = env.reset()
    print('initial observation: {}'.format(observation))
    for t in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, step_info = env.step(action)
        #print('action: {}, observation: {}'.format(action, observation))
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
