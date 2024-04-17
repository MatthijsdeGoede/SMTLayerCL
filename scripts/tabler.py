import numpy as np
import pandas as pd


def tabler(data):
    grouped_data = data.groupby('trial')

    accs = []
    sym_accs = []
    durations = []

    for trial, group in grouped_data:
        last_epoch_data = group[group['epoch'] == group['epoch'].max()]

        acc = last_epoch_data['test_acc'] * 100
        sym_acc = last_epoch_data['test_sym_acc'] * 100
        duration_sum = group['duration'].sum()

        accs.append(acc)
        sym_accs.append(sym_acc)
        durations.append(duration_sum)

    print(
        f"${np.mean(accs):.1f} \pm {np.std(accs):.1f}$ & ${np.mean(sym_accs):.1f} \pm {np.std(sym_accs):.1f}$ & ${np.mean(durations):.0f} \pm {np.std(durations):.0f}$")


tabler(pd.read_csv('../results/20240404181127_100_5.csv'))
tabler(pd.read_csv('../results/20240404194839_100_5_curriculum.csv'))
tabler(pd.read_csv('../results/20240404122410_20_5.csv'))
tabler(pd.read_csv('../results/20240406170143_20_5_curriculum.csv'))
tabler(pd.read_csv('../results/20240404133624_10_10_curriculum.csv'))
tabler(pd.read_csv('../results/20240406160608_5_20_curriculum.csv'))
