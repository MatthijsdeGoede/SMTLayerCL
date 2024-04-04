from mnist_addition import run


def run_experiments():
    run(trials=5, epochs=5, data_fraction=0.2, use_curriculum=False)
    run(trials=5, epochs=5, data_fraction=0.2, use_curriculum=True)
    run(trials=5, epochs=10, data_fraction=0.1, use_curriculum=True)

    run(trials=5, epochs=5, data_fraction=0.5, use_curriculum=False)
    run(trials=5, epochs=5, data_fraction=0.5, use_curriculum=True)
    run(trials=5, epochs=10, data_fraction=0.25, use_curriculum=True)

    run(trials=5, epochs=5, data_fraction=1.0, use_curriculum=False)
    run(trials=5, epochs=5, data_fraction=1.0, use_curriculum=True)
    run(trials=5, epochs=10, data_fraction=0.5, use_curriculum=True)



