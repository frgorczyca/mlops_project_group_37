import pandas as pd
import matplotlib.pyplot as plt


def analyze_label_distribution():
    # Load datasets
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")
    combined_df = pd.concat([train_df, test_df])

    # Create figure with subplots
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Function to analyze and plot distribution
    def plot_distribution(df, title, ax):
        # Get counts and percentages
        counts = df["label"].value_counts().sort_index()
        percentages = (counts / len(df) * 100).round(2)

        # Print statistics
        print(f"\n=== {title} ===")
        print("Counts and Percentages:")
        print(f"Total: {len(df)}")
        print(f"Human (0): {counts[0]} ({percentages[0]}%)")
        print(f"AI (1): {counts[1]} ({percentages[1]}%)")

        # Plot counts
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"{title} (Total: {len(df)})")
        ax.set_xlabel("Label (0=Human, 1=AI)")
        ax.set_ylabel("Count")

        # Add value labels with percentages
        for i, (count, pct) in enumerate(zip(counts, percentages)):
            ax.text(i, count, f"{count}\n({pct}%)", ha="center", va="bottom")

    # Plot distributions
    plot_distribution(combined_df, "Combined Dataset", axes[0])
    plot_distribution(train_df, "Training Dataset", axes[1])
    plot_distribution(test_df, "Testing Dataset", axes[2])

    plt.tight_layout()
    plt.savefig("reports/figures/label_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    analyze_label_distribution()
