from functools import total_ordering
from operator import sub
from tkinter import N

from matplotlib.axis import YAxis
from get_data import concat_dataframe
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use("bmh")

def viz_main(
    combined_dataframe,
    title,
    x_axis,
    y1_label,
    agg_on="timestep",
    groupby_stat="mean",
    y2_label=None,
):
    x_axis = "timestep"  # for graph

    combined_x_axis = agg_on  # groupby this column
    agg_files_stat = "mean"  # max or mean

    val_range = [0, combined_dataframe.shape[0]]

    agg_df = combined_dataframe
    # .groupby(combined_x_axis).agg(agg_files_stat).reset_index()

    agg_df = agg_df.set_index(x_axis)
    X = agg_df.index[val_range[0] : val_range[1]]
    Y = agg_df.iloc[val_range[0] : val_range[1]]

    fig, ax1 = plt.subplots()
    ax1_label = y1_label
    plt.plot(X, Y[ax1_label], color="purple", label=ax1_label, marker="+")
    if y2_label is not None:
        plt.ylim(
            min(agg_df[y1_label].min(), agg_df[y2_label].min()),
            max(agg_df[y1_label].max(), agg_df[y2_label].max()),
        )
    plt.ylabel(ax1_label)
    plt.xlabel(x_axis)
    plt.legend(loc="lower left")

    if y2_label != None:
        ax2 = ax1.twinx()
        ax2_label = y2_label
        ax2.set(ylim=(-2, 100))
        ax2.plot(X, Y[ax2_label], color="orange", label=ax2_label, marker="x")
        plt.ylabel(ax2_label)
        plt.legend(loc="lower right")

    plt.title(title)
    plt.show()


def pandas_scatter_all_cols(data, y_axis="lame_1"):
    for param in data.columns:
        df = data[[param, y_axis]]
        df.plot.scatter(x=param, y=y_axis)
        plt.title(param)
        plt.show()


def pandas_hist_all_cols(data):
    for param in data.columns:
        df = data[param]
        try:
            # df.plot.density()
            df.plot.hist()
        except np.linalg.LinAlgError:
            print("numpy.linalg.LinAlgError: singular matrix")
            df.plot.hist()
        plt.title(param)
        plt.show()


def data_exploration():
    # Why do mass and velocity vars have different shapes?
    # Can we speed up src/solver.cpp::unload_cells
    # Define test suite to create datasets of different start conditions

    sub_df, velocity_df, mass_df = concat_dataframe()

    print(sub_df.describe())
    print(sub_df.head())
    # import pdb; pdb.set_trace()

    pandas_hist_all_cols(sub_df)


if __name__ == "__main__":
    data_exploration()
