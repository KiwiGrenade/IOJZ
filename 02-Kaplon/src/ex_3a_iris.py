import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# --- STAŁE I KONFIGURACJA ---
# ==========================================
SEED = 42
TEST_SIZE = 0.3

# Konfiguracja wykresów
FIG_SIZE_EDA = (8, 6)
FIG_SIZE_CM = (7, 6)
SCATTER_MARKER_SIZE = 80
PALETTE_EDA = "Set1"
CMAP_CM = "Oranges"
PLOT_DPI = 300

# Nazwy cech do wizualizacji EDA (zgodne z nazewnictwem w sklearn)
FEATURE_X = "petal length (cm)"
FEATURE_Y = "petal width (cm)"


# ==========================================
# --- FUNKCJE POMOCNICZE ---
# ==========================================
def plot_eda(df_iris, filename):
    """Generuje i zapisuje wykres analizy cech (EDA) - separowalność gatunków."""
    plt.figure(figsize=FIG_SIZE_EDA)
    
    sns.scatterplot(
        data=df_iris,
        x=FEATURE_X,
        y=FEATURE_Y,
        hue="species",
        palette=PALETTE_EDA,
        s=SCATTER_MARKER_SIZE,
        edgecolor="k"
    )
    
    plt.title("Separowalność gatunków irysów (długość vs szerokość płatka)")
    plt.xlabel("Długość płatka (cm)")
    plt.ylabel("Szerokość płatka (cm)")
    plt.legend(title="Gatunek")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI)
    plt.close()


def train_and_evaluate_gnb(X, y):
    """Dzieli zbiór, trenuje model Gaussian Naive Bayes i zwraca metryki."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    # Używamy wersji GaussianNB, bo cechy to liczby zmiennoprzecinkowe, a nie dyskretne liczności
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    
    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    return y_test, y_pred, acc


def plot_confusion_matrix_heatmap(y_test, y_pred, target_names, filename):
    """Generuje i zapisuje macierz pomyłek w formie mapy ciepła."""
    plt.figure(figsize=FIG_SIZE_CM)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CMAP_CM,
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=False,
        square=True
    )
    
    plt.title("Macierz pomyłek - Klasyfikacja Irysów (Gaussian NB)")
    plt.ylabel("Rzeczywisty gatunek")
    plt.xlabel("Przewidziany gatunek")
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI)
    plt.close()


# ==========================================
# --- GŁÓWNY SKRYPT ---
# ==========================================
def main():
    print("--- ZADANIE 3.A: Naiwny Klasyfikator Bayesa (Irysy) ---")
    
    # 1. Ładowanie danych
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # 2. Generowanie wykresu EDA
    print("Generowanie wykresu analizy cech (EDA)...")
    df_iris = pd.DataFrame(X, columns=iris.feature_names)
    df_iris["species"] = pd.Categorical.from_codes(y, iris.target_names)
    
    eda_file = "iris_eda.png"
    plot_eda(df_iris, eda_file)
    
    # 3. Klasyfikacja
    print("Trenowanie modelu Gaussian Naive Bayes...")
    y_test, y_pred, accuracy = train_and_evaluate_gnb(X, y)
    print(f"Dokładność modelu: {accuracy * 100:.2f}%")
    
    # 4. Generowanie macierzy pomyłek
    print("Generowanie macierzy pomyłek...")
    cm_file = "iris_cm.png"
    plot_confusion_matrix_heatmap(y_test, y_pred, iris.target_names, cm_file)
    
    print(f"\nGotowe! Zapisano wykresy:\n- {eda_file}\n- {cm_file}\n")


if __name__ == "__main__":
    main()
