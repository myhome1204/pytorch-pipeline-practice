from ray import tune

config1 = {
    # "l1": tune.choice([2 ** i for i in range(9)]),
    # "l2": tune.choice([2 ** i for i in range(9)]),
    "lr": tune.loguniform(1e-4, 1e-1),#lr 값의 분포(distribution)
    "batch_size": tune.choice([32,64])
}

config2 = {
    "lr": tune.loguniform(1e-5, 1e-2),
    "batch_size": tune.choice([32, 64]),
}
