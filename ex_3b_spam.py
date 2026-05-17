import io
import os
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# ==========================================
# --- STAŁE I KONFIGURACJA ---
# ==========================================
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
LOCAL_DATASET_FILE = "sms_spam_collection.csv"  # Nazwa pliku do zapisu na dysku
SEED = 42
TEST_SIZE = 0.2
TOP_N_WORDS = 15

# Konfiguracja wykresów
FIG_SIZE_HIST = (9, 5)
FIG_SIZE_CM = (7, 6)
FIG_SIZE_BAR = (9, 6)
PLOT_DPI = 300

# Estetyka (Kolory i palety)
CMAP_CM = "Greens"
COLOR_BAR = "salmon"
PALETTE_HIST = {"ham": "royalblue", "spam": "firebrick"}


# ==========================================
# --- FUNKCJE POMOCNICZE ---
# ==========================================
def load_or_download_spam_data(url, local_filename):
    """
    Sprawdza, czy zbiór danych istnieje lokalnie. Jeśli tak, ładuje go z dysku.
    Jeśli nie, pobiera z sieci, przetwarza w pamięci i zapisuje na dysk.
    """
    if os.path.exists(local_filename):
        print(f"Znaleziono lokalny zbiór danych ('{local_filename}'). Ładowanie z dysku...")
        df = pd.read_csv(local_filename)
        print(f"Załadowano pomyślnie. Zbiór zawiera {len(df)} wiadomości.")
        return df

    print(f"Brak pliku lokalnego. Pobieranie zbioru SMS Spam Collection z {url}...")
    with urllib.request.urlopen(url) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as z:
            with z.open("SMSSpamCollection") as f:
                # Plik w ZIP nie ma nagłówków i jest rozdzielany tabulatorem
                df = pd.read_csv(f, sep="\t", names=["label", "message"])
    
    # Zapisz do standardowego CSV dla przyszłych uruchomień skryptu
    print(f"Zapisywanie zbioru do pliku '{local_filename}'...")
    df.to_csv(local_filename, index=False)
    
    print(f"Pobrano i zapisano pomyślnie. Zbiór zawiera {len(df)} wiadomości.")
    return df


def plot_message_length_distribution(df, filename):
    """Generuje i zapisuje wykres EDA rozkładu długości wiadomości."""
    df_plot = df.copy()
    df_plot["msg_length"] = df_plot["message"].apply(len)

    plt.figure(figsize=FIG_SIZE_HIST)
    sns.histplot(
        data=df_plot,
        x="msg_length",
        hue="label",
        bins=70,
        kde=True,
        palette=PALETTE_HIST,
        alpha=0.6,
    )
    plt.title("Rozkład długości wiadomości: SPAM vs HAM")
    plt.xlabel("Długość wiadomości (liczba znaków)")
    plt.ylabel("Liczba wiadomości")
    plt.xlim(0, 250)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI)
    plt.close()


def train_and_evaluate_mnb(df):
    """Przetwarza tekst, dzieli zbiór, trenuje model MultinomialNB i zwraca wyniki."""
    df["label_bin"] = df["label"].map({"ham": 0, "spam": 1})
    
    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label_bin"], test_size=TEST_SIZE, random_state=SEED
    )

    vectorizer = CountVectorizer(stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    return y_test, y_pred, acc, model, vectorizer


def plot_confusion_matrix_spam(y_test, y_pred, filename):
    """Generuje i zapisuje macierz pomyłek w formie mapy ciepła."""
    plt.figure(figsize=FIG_SIZE_CM)
    cm = confusion_matrix(y_test, y_pred)
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=CMAP_CM,
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        cbar=False,
        square=True
    )
    plt.title("Macierz pomyłek (Rzeczywisty zbiór danych)")
    plt.xlabel("Przewidziana etykieta")
    plt.ylabel("Prawdziwa etykieta")
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI)
    plt.close()


def plot_top_spam_words(model, vectorizer, filename):
    """Identyfikuje najsilniejsze predyktory dla klasy SPAM i rysuje wykres słupkowy."""
    feature_names = vectorizer.get_feature_names_out()
    spam_prob = model.feature_log_prob_[1]
    
    top_indices = np.argsort(spam_prob)[-TOP_N_WORDS:]
    
    top_words = [feature_names[i] for i in top_indices]
    top_probs = [spam_prob[i] for i in top_indices]

    plt.figure(figsize=FIG_SIZE_BAR)
    plt.barh(top_words, top_probs, color=COLOR_BAR, edgecolor="black")
    plt.title(f"Top {TOP_N_WORDS} słów kluczowych determinujących SPAM")
    plt.xlabel("Log-Prawdopodobieństwo (wyższe = ważniejsze)")
    plt.grid(True, axis="x", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=PLOT_DPI)
    plt.close()


# ==========================================
# --- GŁÓWNY SKRYPT ---
# ==========================================
def main():
    print("--- ZADANIE 3.B: Naiwny Klasyfikator Bayesa ---")
    
    # 1. Pobranie i załadowanie danych (lokalnie lub z sieci)
    df = load_or_download_spam_data(DATASET_URL, LOCAL_DATASET_FILE)

    # 2. EDA: Rozkład długości wiadomości
    print("Generowanie wykresu analizy cech (Długość wiadomości)...")
    hist_file = "spam_length_dist.png"
    plot_message_length_distribution(df, hist_file)

    # 3. Klasyfikacja
    print("Trenowanie modelu Multinomial Naive Bayes...")
    y_test, y_pred, accuracy, model, vectorizer = train_and_evaluate_mnb(df)
    print(f"Dokładność modelu: {accuracy * 100:.2f}%")

    # 4. Macierz pomyłek
    print("Generowanie macierzy pomyłek...")
    cm_file = "real_bayes_cm.png"
    plot_confusion_matrix_spam(y_test, y_pred, cm_file)

    # 5. Top najważniejszych słów
    print("Generowanie wykresu: Najczęstsze słowa w spamie...")
    top_words_file = "real_spam_features.png"
    plot_top_spam_words(model, vectorizer, top_words_file)

    print(f"\nGotowe! Zapisano wykresy:\n- {hist_file}\n- {cm_file}\n- {top_words_file}\n")


if __name__ == "__main__":
    main()
