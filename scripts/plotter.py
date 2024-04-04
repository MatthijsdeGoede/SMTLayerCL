import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_acc_per_epoch():
    # Set Seaborn style and color palette
    sns.set_style("darkgrid")
    sns.set_palette("husl")

    # TODO: show 2 plots, one of symbol and one of output correctness, show train and test with/without curriculum in one plot

    # Load CSV into a pandas DataFrame
    data = pd.read_csv('../results/20240404104356_10_10_curriculum.csv')

    # Calculate mean and standard deviation of test_acc for each epoch across trials
    mean_train_acc = data.groupby('epoch')['train_acc'].mean().reset_index()
    std_train_acc = data.groupby('epoch')['train_acc'].std().reset_index()
    mean_train_sym_acc = data.groupby('epoch')['train_sym_acc'].mean().reset_index()
    std_train_sym_acc = data.groupby('epoch')['train_sym_acc'].std().reset_index()

    mean_test_acc = data.groupby('epoch')['test_acc'].mean().reset_index()
    std_test_acc = data.groupby('epoch')['test_acc'].std().reset_index()
    mean_test_sym_acc = data.groupby('epoch')['test_sym_acc'].mean().reset_index()
    std_test_sym_acc = data.groupby('epoch')['test_sym_acc'].std().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=data, x='epoch', y='train_acc', label='Training Output Correctness', marker='o')
    plt.fill_between(mean_train_acc['epoch'], mean_train_acc['train_acc'] - std_train_acc['train_acc'],
                     mean_train_acc['train_acc'] + std_train_acc['train_acc'], alpha=0.2)
    sns.lineplot(data=data, x='epoch', y='train_sym_acc', label='Training Symbol Correctness', marker='o')
    plt.fill_between(mean_train_sym_acc['epoch'],
                     mean_train_sym_acc['train_sym_acc'] - std_train_sym_acc['train_sym_acc'],
                     mean_train_sym_acc['train_sym_acc'] + std_train_sym_acc['train_sym_acc'], alpha=0.2)

    sns.lineplot(data=data, x='epoch', y='test_acc', label='Testing Output Correctness', marker='o')
    plt.fill_between(mean_test_acc['epoch'], mean_test_acc['test_acc'] - std_test_acc['test_acc'],
                     mean_test_acc['test_acc'] + std_test_acc['test_acc'], alpha=0.2)
    sns.lineplot(data=data, x='epoch', y='test_sym_acc', label='Testing Symbol Correctness', marker='o')
    plt.fill_between(mean_test_sym_acc['epoch'], mean_test_sym_acc['test_sym_acc'] - std_test_sym_acc['test_sym_acc'],
                     mean_test_sym_acc['test_sym_acc'] + std_test_sym_acc['test_sym_acc'], alpha=0.2)

    plt.title('Output and Symbol Correctness over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Correctness (%)')
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0))
    plt.grid(True)
    plt.show()


plot_acc_per_epoch()
