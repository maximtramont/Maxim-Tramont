import pandas as pd
from sklearn.preprocessing import StandardScaler


DATA_DIR = "data"

OUTPUT_BRUT = "dataset_cantons_2023_brut.csv"
OUTPUT_ALL_RAW = "dataset_cantons_2023_all_raw.csv"
OUTPUT_SELECTED_RAW = "dataset_cantons_2023_selected_raw.csv"
OUTPUT_SELECTED_NORM = "dataset_cantons_2023_selected_normalized.csv"


SELECTED_COLUMNS = [
    "canton",
    "loyer_moyen_m2",
    "densite_population",
    "taux_logements_vacants",
    "nombre_logements_total",
    "taux_chomage",
    "taux_logements_proprietaire",
]


def load_csv(filename: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/{filename}"
    return pd.read_csv(path)


def compute_taux_chomage(df: pd.DataFrame) -> pd.DataFrame:
    df["taux_chomage"] = (
        df["personnes_au_chomage"]
        / df["personnes_actives"]
    )
    return df.drop(
        columns=["personnes_au_chomage", "personnes_actives"]
    )


def load_and_prepare_sources() -> pd.DataFrame:
    df_loyer = load_csv(
        "loyer_moyen_m2_par_canton_2023_abrege.csv"
    )
    df_densite = load_csv(
        "densite_population_par_canton_2023_abrege.csv"
    )
    df_vacants = load_csv(
        "logements_vacants_par_canton_2023_abrege.csv"
    )
    df_logements = load_csv(
        "nombre_de_logement_2023.csv"
    )
    df_chomage = load_csv(
        "personnes_au_chomage_par_canton_2023.csv"
    )
    df_proprietaire = load_csv(
        "logements_occupes_selon_le_statut_occupation.csv"
    )

    df_chomage = compute_taux_chomage(df_chomage)

    df_proprietaire = df_proprietaire[
        ["canton", "taux_logements_proprietaire"]
    ]

    df = df_loyer.merge(df_densite, on="canton", how="inner")
    df = df.merge(df_vacants, on="canton", how="inner")
    df = df.merge(df_logements, on="canton", how="inner")
    df = df.merge(df_chomage, on="canton", how="inner")
    df = df.merge(df_proprietaire, on="canton", how="inner")

    return df


def save_selected_and_normalized(df: pd.DataFrame) -> None:
    df_selected = df[SELECTED_COLUMNS]
    df_selected.to_csv(OUTPUT_SELECTED_RAW, index=False)

    features = df_selected.drop(columns=["canton"])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(
        scaled,
        columns=features.columns
    )
    df_scaled.insert(0, "canton", df_selected["canton"])

    df_scaled.to_csv(OUTPUT_SELECTED_NORM, index=False)


def main() -> None:
    df_brut = load_and_prepare_sources()

    df_brut.to_csv(OUTPUT_BRUT, index=False)
    df_brut.to_csv(OUTPUT_ALL_RAW, index=False)

    save_selected_and_normalized(df_brut)


if __name__ == "__main__":
    main()
