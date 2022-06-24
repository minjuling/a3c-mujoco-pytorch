class Config(object):
    mode = 'train'
    env_name = 'Hopper-v1'
    # space: (state, action)
    # InvertedPendulum-v1 (4,3)
    # Swimmer-v4 (8,2)
    # Hopper-v1 (11,3)
    # Ant-v4 (27, 8)
    # Humanoid-v4 (376, 17)
    cuda_num = None # no cuda -> None
    num_processes = 2
    lr = 1e-4
    memory_size = 100000
    epsilon = 1
    epsilon_min = 0.01
    eps_decay = 30000
    use_cuda = True
    save_frequency = 300
    loss_frequency = 500
    test_reward_frequency = 100
    batch_size = 32
    initial_observe_episode = 100
    logs_path = './logs'
    ckpt_path = './results'
    gamma = 0.99
    tau = 1
    seed = 1004
    num_steps = 20
    max_step_len = 10000