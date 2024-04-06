import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_acc_for_samples(df, df_cl):
    # Set Seaborn style and color palette
    sns.set_style("darkgrid")

    # To correct for the batch size
    df['samples'] = df['samples'] * 128
    df_cl['samples'] = df_cl['samples'] * 128

    # Calculate mean and standard deviation of test_acc for each epoch across trials
    mean_test_with = df.groupby('epoch')['test_acc'].mean().reset_index()
    std_test_with = df.groupby('epoch')['test_acc'].std().reset_index()
    mean_test_without = df_cl.groupby('epoch')['test_acc'].mean().reset_index()
    std_test_without = df_cl.groupby('epoch')['test_acc'].std().reset_index()

    # Calculate mean and standard deviation of test_sym_acc for each epoch across trials
    mean_test_sym_with = df.groupby('epoch')['test_sym_acc'].mean().reset_index()
    std_test_sym_with = df.groupby('epoch')['test_sym_acc'].std().reset_index()
    mean_test_sym_without = df_cl.groupby('epoch')['test_sym_acc'].mean().reset_index()
    std_test_sym_without = df_cl.groupby('epoch')['test_sym_acc'].std().reset_index()

    plt.figure(figsize=(10, 6))

    # Plotting the lines for the output correctness
    sns.lineplot(data=df, x='samples', y='test_acc', label='Output Correctness without CL', marker='o')
    plt.fill_between(mean_test_without['epoch'], mean_test_without['test_acc'] - std_test_without['test_acc'],
                     mean_test_without['test_acc'] + std_test_without['test_acc'], alpha=0.2)

    sns.lineplot(data=df_cl, x='samples', y='test_acc', label='Output Correctness with CL', marker='o')
    plt.fill_between(mean_test_with['epoch'], mean_test_with['test_acc'] - std_test_with['test_acc'],
                     mean_test_with['test_acc'] + std_test_with['test_acc'], alpha=0.2)

    # Plotting the lines for symbol correctness
    sns.lineplot(data=df, x='samples', y='test_sym_acc', label='Symbol Correctness without CL', marker='o')
    plt.fill_between(mean_test_sym_without['epoch'], mean_test_sym_without['test_sym_acc'] - std_test_sym_without['test_sym_acc'],
                     mean_test_sym_without['test_sym_acc'] + std_test_sym_without['test_sym_acc'], alpha=0.2)

    sns.lineplot(data=df_cl, x='samples', y='test_sym_acc', label='Symbol Correctness with CL', marker='o')
    plt.fill_between(mean_test_sym_with['epoch'], mean_test_sym_with['test_sym_acc'] - std_test_sym_with['test_sym_acc'],
                     mean_test_sym_with['test_sym_acc'] + std_test_sym_with['test_sym_acc'], alpha=0.2)

    plt.title('Output and Symbol Correctness on Test Set vs Number of Samples Seen During Training')
    plt.xlabel('Number of Training Samples')
    plt.ylabel('Correctness (%)')
    plt.legend(loc='upper left')
    plt.grid(True)
    start_plot = df_cl['samples'].min() - 10
    end_plot = df_cl['samples'].max() + 10
    plt.xlim(start_plot, end_plot)
    plt.show()


def plot_acc_for_pairs(df_cl):
    # Set Seaborn style and color palette
    sns.set_style("darkgrid")

    # Calculate mean and standard deviation of test_acc for each epoch across trials
    mean_test_with = df_cl.groupby('epoch')['test_acc'].mean().reset_index()
    std_test_with = df_cl.groupby('epoch')['test_acc'].std().reset_index()

    # Calculate mean and standard deviation of test_sym_acc for each epoch across trials
    mean_test_sym_with = df_cl.groupby('epoch')['test_sym_acc'].mean().reset_index()
    std_test_sym_with = df_cl.groupby('epoch')['test_sym_acc'].std().reset_index()

    plt.figure(figsize=(10, 6))

    # Plotting the line for the output correctness
    sns.lineplot(data=df_cl, x='pairs', y='test_acc', label='Output Correctness with CL', marker='o')
    plt.fill_between(mean_test_with['epoch'], mean_test_with['test_acc'] - std_test_with['test_acc'],
                     mean_test_with['test_acc'] + std_test_with['test_acc'], alpha=0.2)

    # Plotting the line for symbol correctness
    sns.lineplot(data=df_cl, x='pairs', y='test_sym_acc', label='Symbol Correctness with CL', marker='o')
    plt.fill_between(mean_test_sym_with['epoch'], mean_test_sym_with['test_sym_acc'] - std_test_sym_with['test_sym_acc'],
                     mean_test_sym_with['test_sym_acc'] + std_test_sym_with['test_sym_acc'], alpha=0.2)

    plt.title('Output and Symbol Correctness on Test Set vs Percentage of Pairs Seen During Training')
    plt.xlabel('Pairs Seen (%)')
    plt.ylabel('Correctness (%)')
    plt.legend(loc='upper left')
    plt.grid(True)
    start_plot = df_cl['pairs'].min() - 10
    end_plot = df_cl['pairs'].max() + 10
    plt.xlim(start_plot, end_plot)
    plt.show()


without_cl = pd.read_csv('../results/20240404181127_100_5.csv')
with_cl = pd.read_csv('../results/20240404194839_100_5_curriculum.csv')

# Plot the test output and symbolic accuracy against the number of samples
plot_acc_for_samples(without_cl, with_cl)
# Plot the test output and symbolic accuracy against the percentage of seen pairs
plot_acc_for_pairs(with_cl)