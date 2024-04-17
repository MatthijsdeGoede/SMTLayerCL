from mnist_addition import run


def run_experiments():
    # BASE_20_5
    run(trials=5, epochs=5, data_fraction=0.2, use_curriculum=False)
    # CL_20_5
    run(trials=5, epochs=5, data_fraction=0.2, use_curriculum=True)
    # CL_10_10
    run(trials=5, epochs=10, data_fraction=0.1, use_curriculum=True)
    # CL_5_20
    run(trials=5, epochs=20, data_fraction=0.05, use_curriculum=True)

    # BASE_100_5
    run(trials=5, epochs=5, data_fraction=1.0, use_curriculum=False)
    # CL_100_5
    run(trials=5, epochs=5, data_fraction=1.0, use_curriculum=True)


if __name__ == '__main__':
    run_experiments()
