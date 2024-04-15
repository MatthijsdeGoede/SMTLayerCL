import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def format_func(value, tick_number):
    # Format the value with comma separator for thousands
    return '{:,.0f}'.format(value)

def plot_acc_for_samples_multiple():
    no_cl = pd.read_csv('../results/20240404122410_20_5.csv')
    cl_20_5 = pd.read_csv('../results/20240406170143_20_5_curriculum.csv')
    cl_10_10 = pd.read_csv('../results/20240404133624_10_10_curriculum.csv')
    cl_5_20 = pd.read_csv('../results/20240406160608_5_20_curriculum.csv')

    # Set Seaborn style and color palette
    sns.set_style("darkgrid")

    # To correct for the batch size
    no_cl['samples'] = no_cl['samples'] * 128
    cl_20_5['samples'] = cl_20_5['samples'] * 128
    cl_10_10['samples'] = cl_10_10['samples'] * 128
    cl_5_20['samples'] = cl_5_20['samples'] * 128

    # Calculate mean and standard deviation of test_sym_acc for each epoch across trials
    mean_test_sym_no = no_cl.groupby('epoch')['test_sym_acc'].mean().reset_index()
    std_test_sym_no = no_cl.groupby('epoch')['test_sym_acc'].std().reset_index()
    mean_test_sym_20_5 = cl_20_5.groupby('epoch')['test_sym_acc'].mean().reset_index()
    std_test_sym_20_5 = cl_20_5.groupby('epoch')['test_sym_acc'].std().reset_index()
    mean_test_sym_10_10 = cl_10_10.groupby('epoch')['test_sym_acc'].mean().reset_index()
    std_test_sym_10_10 = cl_10_10.groupby('epoch')['test_sym_acc'].std().reset_index()
    mean_test_sym_5_20 = cl_5_20.groupby('epoch')['test_sym_acc'].mean().reset_index()
    std_test_sym_5_20 = cl_5_20.groupby('epoch')['test_sym_acc'].std().reset_index()

    # Plotting the lines for symbol correctness
    sns.lineplot(data=no_cl, x='samples', y='test_sym_acc', label='SC BASE_20_5', marker='o')
    plt.fill_between(mean_test_sym_no['epoch'], mean_test_sym_no['test_sym_acc'] - std_test_sym_no['test_sym_acc'],
                     mean_test_sym_no['test_sym_acc'] + std_test_sym_no['test_sym_acc'], alpha=0.2)

    sns.lineplot(data=cl_20_5, x='samples', y='test_sym_acc', label='SC CL_20_5', marker='o')
    plt.fill_between(mean_test_sym_20_5['epoch'], mean_test_sym_20_5['test_sym_acc'] - std_test_sym_20_5['test_sym_acc'],
                     mean_test_sym_20_5['test_sym_acc'] + std_test_sym_20_5['test_sym_acc'], alpha=0.2)

    sns.lineplot(data=cl_10_10, x='samples', y='test_sym_acc', label='SC CL_10_10', marker='o')
    plt.fill_between(mean_test_sym_10_10['epoch'], mean_test_sym_10_10['test_sym_acc'] - std_test_sym_10_10['test_sym_acc'],
                     mean_test_sym_10_10['test_sym_acc'] + std_test_sym_10_10['test_sym_acc'], alpha=0.2)

    sns.lineplot(data=cl_5_20, x='samples', y='test_sym_acc', label='SC CL_5_20', marker='o')
    plt.fill_between(mean_test_sym_5_20['epoch'], mean_test_sym_5_20['test_sym_acc'] - std_test_sym_5_20['test_sym_acc'],
                     mean_test_sym_5_20['test_sym_acc'] + std_test_sym_5_20['test_sym_acc'], alpha=0.2)

    plt.xlabel('Number of Training Samples')
    plt.ylabel('Correctness (%)')
    plt.legend(loc='upper left')
    plt.grid(True)
    start_plot = cl_5_20['samples'].min() - 10
    end_plot = cl_5_20['samples'].max() + 10
    plt.xlim(start_plot, end_plot)
    # Apply the custom formatter to the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    plt.show()



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
    sns.lineplot(data=df, x='samples', y='test_acc', label='OC BASE_20_5', marker='o')
    plt.fill_between(mean_test_without['epoch'], mean_test_without['test_acc'] - std_test_without['test_acc'],
                     mean_test_without['test_acc'] + std_test_without['test_acc'], alpha=0.2)

    sns.lineplot(data=df_cl, x='samples', y='test_acc', label='OC CL_20_5', marker='o')
    plt.fill_between(mean_test_with['epoch'], mean_test_with['test_acc'] - std_test_with['test_acc'],
                     mean_test_with['test_acc'] + std_test_with['test_acc'], alpha=0.2)

    # Plotting the lines for symbol correctness
    sns.lineplot(data=df, x='samples', y='test_sym_acc', label='SC BASE_20_5', marker='o')
    plt.fill_between(mean_test_sym_without['epoch'], mean_test_sym_without['test_sym_acc'] - std_test_sym_without['test_sym_acc'],
                     mean_test_sym_without['test_sym_acc'] + std_test_sym_without['test_sym_acc'], alpha=0.2)

    sns.lineplot(data=df_cl, x='samples', y='test_sym_acc', label='SC CL_20_5', marker='o')
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
    # Apply the custom formatter to the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
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
    sns.lineplot(data=df_cl, x='pairs', y='test_acc', label='OC CL_100_5', marker='o')
    plt.fill_between(mean_test_with['epoch'], mean_test_with['test_acc'] - std_test_with['test_acc'],
                     mean_test_with['test_acc'] + std_test_with['test_acc'], alpha=0.2)

    # Plotting the line for symbol correctness
    sns.lineplot(data=df_cl, x='pairs', y='test_sym_acc', label='SC CL_100_5', marker='o')
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
    # Apply the custom formatter to the x-axis
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
    plt.show()

without_cl = pd.read_csv('../results/20240404122410_20_5.csv')
with_cl = pd.read_csv('../results/20240406170143_20_5_curriculum.csv')

# Plot the test output and symbolic accuracy against the number of samples
plot_acc_for_samples(without_cl, with_cl)
# Plot the test output and symbolic accuracy against the percentage of seen pairs
#plot_acc_for_pairs(with_cl)
# Plot the symbolic accuracy against number of samples for each of the methods
plot_acc_for_samples_multiple()