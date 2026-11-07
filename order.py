
"""
Lab 05 - Order Task

Compare three ways to give a Transformer position information.
"""

import csv

from lab05_starter import train_order
from attention import MultiHeadAttention


def main():
    results = []

    position_types = ["none", "learned", "sinusoidal"]
    seeds = [0, 1, 2]

    for position in position_types:
        for seed in seeds:

            # Train on sequences of length 16.
            # Test on lengths 16 and 32.
            accuracies = train_order(
                pos=position,
                seed=seed,
                steps=800,
                train_len=16,
                test_lens=(16, 32),
                attention_cls=MultiHeadAttention
            )

            for length, accuracy in zip([16, 32], accuracies):
                print(
                    "Position:", position,
                    "| Seed:", seed,
                    "| Test length:", length,
                    "| Accuracy:", round(accuracy, 3)
                )

                results.append([
                    position, seed, length, accuracy, "fixed"
                ])

    # Also test training with different sequence lengths.
    for position in ["learned", "sinusoidal"]:
        for seed in seeds:

            accuracies = train_order(
                pos=position,
                seed=seed,
                steps=800,
                mixed_lens=(8, 16, 24),
                test_lens=(16, 32),
                attention_cls=MultiHeadAttention
            )

            for length, accuracy in zip([16, 32], accuracies):
                print(
                    "Mixed training | Position:", position,
                    "| Seed:", seed,
                    "| Test length:", length,
                    "| Accuracy:", round(accuracy, 3)
                )

                results.append([
                    position, seed, length, accuracy, "mixed"
                ])

    # Save all results.
    with open("order_results.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "position", "seed", "test_length",
            "accuracy", "training_type"
        ])

        writer.writerows(results)

    print("Results saved to order_results.csv")


if __name__ == "__main__":
    main()
