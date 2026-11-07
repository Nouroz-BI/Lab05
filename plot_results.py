
"""
Lab 05 - Plot results

Create charts from the saved CSV files.
"""

import pandas as pd
import matplotlib.pyplot as plt


def plot_recall():
    data = pd.read_csv("recall_results.csv")

    # Average the three seeds for each model and sequence length.
    average = data.groupby(
        ["model", "sequence_length"]
    )["accuracy"].mean().reset_index()

    for model_name in average["model"].unique():
        model_data = average[average["model"] == model_name]

        plt.plot(
            model_data["sequence_length"],
            model_data["accuracy"],
            marker="o",
            label=model_name
        )

    plt.xlabel("Sequence length")
    plt.ylabel("Average accuracy")
    plt.title("Recall: Attention vs LSTM")
    plt.legend()
    plt.tight_layout()
    plt.savefig("recall_plot.png")
    plt.show()


def plot_order():
    data = pd.read_csv("order_results.csv")

    # Average results across seeds.
    average = data.groupby(
        ["position", "training_type", "test_length"]
    )["accuracy"].mean().reset_index()

    for (position, training_type), group in average.groupby(
        ["position", "training_type"]
    ):
        group = group.sort_values("test_length")

        label = position + " - " + training_type

        plt.plot(
            group["test_length"],
            group["accuracy"],
            marker="o",
            label=label
        )

    plt.xlabel("Test sequence length")
    plt.ylabel("Average accuracy")
    plt.title("Order Task Results")
    plt.legend()
    plt.tight_layout()
    plt.savefig("order_plot.png")
    plt.show()


def plot_timing():
    data = pd.read_csv("timing_results.csv")

    plt.plot(
        data["sequence_length"],
        data["lstm_seconds"],
        marker="o",
        label="LSTM"
    )

    plt.plot(
        data["sequence_length"],
        data["attention_seconds"],
        marker="o",
        label="Attention"
    )

    plt.xlabel("Sequence length")
    plt.ylabel("Seconds per training step")
    plt.title("Runtime Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("timing_plot.png")
    plt.show()


if __name__ == "__main__":
    plot_recall()
    plot_order()
    plot_timing()

    print("Charts saved as PNG files.")
