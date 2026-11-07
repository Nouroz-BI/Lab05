
"""
Lab 05 - Model depth and timing

Compare normalization settings and measure runtime.
"""

import csv
import time
import torch
import torch.nn as nn

from lab05_starter import train_order, OrderClassifier
from attention import MultiHeadAttention


def compare_models():
    """Compare normalization and learning-rate warmup."""

    settings = [
        ("post", 0),
        ("post", 100),
        ("pre", 0)
    ]

    results = []

    for norm_type, warmup_steps in settings:

        accuracies = train_order(
            pos="sinusoidal",
            seed=0,
            depth=12,
            norm=norm_type,
            warmup=warmup_steps,
            steps=800,
            test_lens=(16, 32),
            attention_cls=MultiHeadAttention
        )

        print(
            "Norm:", norm_type,
            "| Warmup:", warmup_steps,
            "| Accuracy at 16:", round(accuracies[0], 3),
            "| Accuracy at 32:", round(accuracies[1], 3)
        )

        results.append([
            norm_type, warmup_steps,
            accuracies[0], accuracies[1]
        ])

    with open("stack_results.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "normalization", "warmup",
            "accuracy_length_16", "accuracy_length_32"
        ])

        writer.writerows(results)


class LSTMOutput(nn.Module):
    """Return only the output sequence from the LSTM."""

    def __init__(self, input_size=32, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, batch_first=True
        )

    def forward(self, x):
        output, _ = self.lstm(x)
        return output


def measure_time(model, x, repeats=20):
    """Measure the average forward and backward pass time."""

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Run once before measuring.
    output = model(x)
    loss = output.square().mean()
    loss.backward()
    optimizer.zero_grad()

    start = time.perf_counter()

    for _ in range(repeats):
        output = model(x)
        loss = output.square().mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    end = time.perf_counter()

    return (end - start) / repeats


def compare_timing():
    """Compare an LSTM with one attention layer."""

    results = []
    sequence_lengths = [100, 200, 400, 800]

    for length in sequence_lengths:
        x = torch.randn(8, length, 32)

        lstm = LSTMOutput()
        attention = MultiHeadAttention(32, 4)

        lstm_time = measure_time(lstm, x)
        attention_time = measure_time(attention, x)

        print(
            "Length:", length,
            "| LSTM seconds:", round(lstm_time, 4),
            "| Attention seconds:", round(attention_time, 4)
        )

        results.append([length, lstm_time, attention_time])

    with open("timing_results.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "sequence_length", "lstm_seconds", "attention_seconds"
        ])

        writer.writerows(results)


def main():
    compare_models()
    compare_timing()

    print("Part 4 results saved.")


if __name__ == "__main__":
    main()
