import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_f1_score(
    data: pd.DataFrame, title: str, x_col: str, y_col: str, filename: str
):

    data_copy = data.copy(deep=True)
    data_copy[x_col] = data[x_col].astype(float)

    plt.figure()
    sns.boxplot(data=data_copy, x=x_col, y=y_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_f1_score_line_plot(
    data: pd.DataFrame,
    title: str,
    x_col: str,
    y_col: str,
    hue: str,
    filename: str,
):

    plt.title(title)
    sns.lineplot(data=data, x=x_col, y=y_col, hue=hue)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()