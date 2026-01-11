import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


DATA_PATH = "result/dataset_cantons_2023_selected_normalized.csv"

FEATURES = [
    "densite_population",
    "taux_logements_vacants",
    "nombre_logements_total",
    "taux_chomage",
    "taux_logements_proprietaire",
]

TARGET = "loyer_moyen_m2"


def plot_loyer_distribution(df):
    plt.figure(figsize=(6, 4))
    sns.histplot(df[TARGET], kde=True)
    plt.title("Distribution du loyer moyen au m² (standardisé)")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[TARGET])
    plt.title("Boxplot du loyer moyen au m²")
    plt.tight_layout()
    plt.show()


def plot_scatter_vs_target(df):
    for feature in FEATURES:
        plt.figure(figsize=(5, 4))
        sns.regplot(
            x=df[feature],
            y=df[TARGET],
            ci=None,
            scatter_kws={"alpha": 0.7}
        )
        plt.xlabel(feature)
        plt.ylabel("Loyer moyen au m²")
        plt.title(f"Loyer moyen au m² vs {feature}")
        plt.tight_layout()
        plt.show()


def plot_correlation_matrix(df):
    cols = [TARGET] + FEATURES
    corr = df[cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True
    )
    plt.title("Matrice de corrélation (variables standardisées)")
    plt.tight_layout()
    plt.show()


def main():
    df = pd.read_csv(DATA_PATH)

    plot_loyer_distribution(df)
    plot_scatter_vs_target(df)
    plot_correlation_matrix(df)


if __name__ == "__main__":
    main()
