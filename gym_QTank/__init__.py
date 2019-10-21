from gym.envs.registration import register

register(
    id='QTank-v0',
    entry_point='gym_QTank.envs:QTankEnv',
)
