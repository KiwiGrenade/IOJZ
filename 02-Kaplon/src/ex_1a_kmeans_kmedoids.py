import io
import urllib.request
import zipfile

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    pairwise_distances,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn_extra.cluster import KMedoids

# Wymusza zapis do pliku bez otwierania okien - zapobiega zawieszaniu się na Macach
matplotlib.use("Agg")

# ==========================================
# --- STAŁE I KONFIGURACJA (Brak Magic Numbers) ---
# ==========================================

# Hiperparametry algorytmów grupowania
SEEDS = [101]
N_CLUSTERS = 3
MAX_ITERATIONS = 100
N_INITS = 1

# Zakresy dla analizy liczby klastrów (Metoda Łokcia i Silhouette)
K_MIN_ELBOW = 1
K_MIN_SILHOUETTE = 2
K_MAX_ANALYSIS = 10  # Wartość ekskluzywna dla zakresu range()

# Parametry redukcji wymiarowości (PCA)
PCA_COMPONENTS = 2

# Ustawienia wizualizacji i wykresów
FIG_SIZE = (15, 5)
TITLE_FONT_SIZE = 14
PLOT_DPI = 300


# ==========================================
# --- FUNKCJE RYSUJĄCE ---
# ==========================================


def getClusterPlots(data, seed, labels_kmeans, labels_kmedoids):
    """Generuje i zapisuje wykres porównawczy klastrów K-means vs K-medoids po redukcji PCA."""
    file_name = f"clustering_{seed}.png"
    print(f"Generowanie wykresów z wynikami grupowania: {file_name}...")

    pca = PCA(n_components=PCA_COMPONENTS)
    data_pca = pca.fit_transform(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)

    # Wykres K-means
    ax1.scatter(
        data_pca[:, 0],
        data_pca[:, 1],
        c=labels_kmeans,
        cmap="viridis",
        edgecolor="k",
    )
    ax1.set_title(f"Grupowanie K-means dla ziarna {seed}")

    # Wykres K-medoids
    ax2.scatter(
        data_pca[:, 0],
        data_pca[:, 1],
        c=labels_kmedoids,
        cmap="viridis",
        edgecolor="k",
    )
    ax2.set_title(f"Grupowanie K-medoids dla ziarna {seed}")

    plt.tight_layout()
    plt.savefig(file_name, dpi=PLOT_DPI)
    plt.close()


def getElbowAndSilhouettePlots(data, seed, n_inits):
    """Generuje i zapisuje połączony wykres analizy Łokcia oraz wskaźnika Silhouette."""
    file_name = f"clustering_analysis_seed_{seed}.png"
    print(f"Generowanie wykresów z metodą łokcia i Silhouette: {file_name}...")

    # --- 1. Obliczenia dla Metody Łokcia ---
    inertias = []
    k_range_elbow = range(K_MIN_ELBOW, K_MAX_ANALYSIS)
    for k in k_range_elbow:
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_inits)
        km.fit(data)
        inertias.append(km.inertia_)

    # --- 2. Obliczenia dla wskaźnika Silhouette ---
    sil_scores = []
    k_range_sil = range(K_MIN_SILHOUETTE, K_MAX_ANALYSIS)
    for k in k_range_sil:
        km = KMeans(n_clusters=k, random_state=seed, n_init=n_inits)
        labels = km.fit_predict(data)
        sil_scores.append(silhouette_score(data, labels))

    # --- 3. Tworzenie wspólnego wykresu (1 wiersz, 2 kolumny) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIG_SIZE)

    # Wykres po lewej: Metoda Łokcia
    ax1.plot(k_range_elbow, inertias, marker="o", linestyle="--", color="b")
    ax1.set_title("Metoda Łokcia (Elbow Method)")
    ax1.set_xlabel("Liczba klastrów (k)")
    ax1.set_ylabel("Suma kwadratów odległości (Inertia)")
    ax1.set_xticks(k_range_elbow)
    ax1.grid(True, linestyle=":", alpha=0.7)

    # Wykres po prawej: Silhouette
    ax2.plot(
        k_range_sil, sil_scores, marker="s", linestyle="-", color="purple"
    )
    ax2.set_title("Analiza wskaźnika Silhouette")
    ax2.set_xlabel("Liczba klastrów (k)")
    ax2.set_ylabel("Średni wynik Silhouette (wyżej = lepiej)")
    ax2.set_xticks(k_range_sil)
    ax2.grid(True, linestyle=":", alpha=0.7)

    # Wspólny tytuł dla całego obrazka
    plt.suptitle(
        f"Analiza doboru liczby klastrów dla zbioru Wine i ziarna {seed}",
        fontsize=TITLE_FONT_SIZE,
        fontweight="bold",
    )

    plt.tight_layout()
    plt.savefig(file_name, dpi=PLOT_DPI)
    plt.close()


# ==========================================
# --- GŁÓWNY SKRYPT URUCHOMIENIOWY ---
# ==========================================


def main():
    print("--- ZADANIE 1.A: Grupowanie ---")

    # Ładowanie danych
    wine = datasets.load_wine()
    data = wine.data

    # Iteracja po zdefiniowanych ziarnach losowości
    for seed in SEEDS:
        # K-means
        kmeans = KMeans(
            n_clusters=N_CLUSTERS,
            random_state=seed,
            n_init=N_INITS,
            max_iter=MAX_ITERATIONS,
        )
        labels_kmeans = kmeans.fit_predict(data)

        # K-medoids
        kmedoids = KMedoids(
            n_clusters=N_CLUSTERS,
            random_state=seed,
            init="random",
            max_iter=MAX_ITERATIONS,
        )
        labels_kmedoids = kmedoids.fit_predict(data)

        # Generowanie raportów graficznych
        getClusterPlots(data, seed, labels_kmeans, labels_kmedoids)
        getElbowAndSilhouettePlots(data, seed, N_INITS)


if __name__ == "__main__":
    main()
