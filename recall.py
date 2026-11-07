
"""
Lab 05 - Recall Task

Compare attention and LSTM on different sequence lengths.
"""

import csv

from lab05_starter import train_recall
from attention import scaled_dot_product_attention


def main():
    results = []

    # Test short, medium and long sequences.
    sequence_lengths = [30, 100, 400]
    models = ["attention", "lstm"]
    seeds = [0, 1, 2]

    for length in sequence_lengths:
        for model_name in models:
            for seed in seeds:

                # Use our attention function for the attention model.
                if model_name == "attention":
                    accuracy, first_weight = train_recall(
                        kind="attention",
                        seq_len=length,
                        seed=seed,
                        attention_fn=scaled_dot_product_attention
                    )
                else:
                    accuracy, first_weight = train_recall(
                        kind="lstm",
                        seq_len=length,
                        seed=seed
                    )

                print(
                    "Length:", length,
                    "| Model:", model_name,
                    "| Seed:", seed,
                    "| Accuracy:", round(accuracy, 3),
                    "| First-step weight:", first_weight
                )

                results.append([
                    length, model_name, seed, accuracy, first_weight
                ])

    # Save the results so they can be used in plots and the report.
    with open("recall_results.csv", "w", newline="") as file:
        writer = csv.writer(file)

        writer.writerow([
            "sequence_length", "model", "seed",
            "accuracy", "first_step_weight"
        ])

        writer.writerows(results)

    print("Results saved to recall_results.csv")


if __name__ == "__main__":
    main()
