import io
import urllib.request
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- STAŁE (Brak "Magic Numbers") ---
X_MIN = 0.0
X_MAX = np.pi
Y_MIN = 0.0
Y_MAX = 1.2  # Bezpieczny margines powyżej szczytu sinusa (1.0)
THEORETICAL_INTEGRAL = 2.0

# Wspólne ziarno losowości dla powtarzalności wyników
SEED = 42

# Wartości N dla głównych symulacji oraz histogramu
N_VALUES_MAIN = [10000, 100000, 1000000]

# Parametry dla histogramu
N_SIMULATIONS = 100  # Liczba powtórzeń symulacji dla każdego N
HIST_BINS = 20

# Parametry analizy zbieżności (generowane logarytmicznie)
N_CONVERGENCE_SAMPLES = 50

# Rozdzielczość rysowania linii funkcji f(x)
LINE_RESOLUTION = 100

# Wymiary wykresów
FIG_SIZE_COMBINED = (18, 5)
FIG_SIZE_CONVERGENCE = (9, 5)
FIG_SIZE_HISTOGRAM = (10, 6)


def target_function(x):
    """Funkcja, pod którą liczymy całkę."""
    return np.sin(x)


def run_monte_carlo(n_points, rng):
    """Przeprowadza symulację Monte Carlo przy użyciu przekazanego generatora rng."""
    x_mc = rng.uniform(X_MIN, X_MAX, n_points)
    y_mc = rng.uniform(Y_MIN, Y_MAX, n_points)

    y_actual = target_function(x_mc)
    points_under = y_mc <= y_actual

    rectangle_area = (X_MAX - X_MIN) * (Y_MAX - Y_MIN)
    calculated_integral = (np.sum(points_under) / n_points) * rectangle_area

    return x_mc, y_mc, points_under, calculated_integral


def plot_combined_distributions(results_dict, filename):
    """Generuje jeden wspólny plik graficzny zawierający 3 podwykresy rozkładu punktów."""
    fig, axes = plt.subplots(1, len(N_VALUES_MAIN), figsize=FIG_SIZE_COMBINED)

    x_line = np.linspace(X_MIN, X_MAX, LINE_RESOLUTION)
    y_line = target_function(x_line)

    for idx, n in enumerate(N_VALUES_MAIN):
        ax = axes[idx]
        x_mc, y_mc, points_under, _ = results_dict[n]

        marker_size = 1.0 if n < 1000000 else 0.1
        alpha_val = 0.6 if n < 1000000 else 0.2

        ax.scatter(
            x_mc[points_under],
            y_mc[points_under],
            color="green",
            s=marker_size,
            alpha=alpha_val,
            label="Pod wykresem" if idx == 0 else "",
        )
        ax.scatter(
            x_mc[~points_under],
            y_mc[~points_under],
            color="red",
            s=marker_size,
            alpha=alpha_val,
            label="Nad wykresem" if idx == 0 else "",
        )

        ax.plot(
            x_line,
            y_line,
            color="black",
            linewidth=2,
            label="f(x) = sin(x)" if idx == 0 else "",
        )

        ax.set_title(f"Rozkład punktów dla N = {n:,}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(filename, dpi=200)
    plt.close()


def simulate_convergence(seed):
    """Oblicza wartości całki dla rosnącej logarytmicznie liczby punktów."""
    rng = np.random.RandomState(seed)
    n_values = np.logspace(2, 6, num=N_CONVERGENCE_SAMPLES, dtype=int)
    n_values = np.unique(np.sort(np.concatenate((n_values, N_VALUES_MAIN))))

    mc_results = []
    for n in n_values:
        # Resetujemy stan dla każdego n, aby mniejsze N brało podzbiór punktów większego N
        rng_local = np.random.RandomState(seed)
        _, _, _, area = run_monte_carlo(n, rng_local)
        mc_results.append(area)

    return n_values, mc_results


def plot_convergence(n_values, mc_results, main_results, filename):
    """Generuje wykres zbieżności w osobnym pliku z zaznaczonymi punktami węzłowymi."""
    plt.figure(figsize=FIG_SIZE_CONVERGENCE)
    plt.plot(
        n_values, mc_results, label="Ścieżka symulacji", color="blue", alpha=0.7
    )
    plt.axhline(
        y=THEORETICAL_INTEGRAL,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Wartość teoretyczna = {THEORETICAL_INTEGRAL}",
    )

    for n in N_VALUES_MAIN:
        _, _, _, calculated_integral = main_results[n]
        plt.axvline(x=n, color="gray", linestyle="--", alpha=0.5)
        plt.scatter(
            n,
            calculated_integral,
            color="gold",
            edgecolor="black",
            s=120,
            marker="*",
            zorder=5,
            label=f"Wynik dla N={n:,}: {calculated_integral:.4f}",
        )

    plt.xscale("log")
    plt.title("Zbieżność metody Monte Carlo (Skala Logarytmiczna)")
    plt.xlabel("Liczba losowanych punktów (N)")
    plt.ylabel("Obliczona wartość całki")
    plt.legend(loc="lower left")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def generate_and_plot_histograms(seed, filename):
    """Przeprowadza wielokrotne symulacje dla różnych N i generuje porównawczy histogram."""
    print(
        f"Uruchamianie {N_SIMULATIONS} symulacji dla każdego N w celu wygenerowania histogramu..."
    )
    plt.figure(figsize=FIG_SIZE_HISTOGRAM)

    # Kolory dla poszczególnych wielkości N
    colors = {10000: "teal", 100000: "royalblue", 1000000: "purple"}

    # Wspólny generator nadrzędny zachowujący powtarzalność eksperymentu
    rng = np.random.RandomState(seed)

    for n in N_VALUES_MAIN:
        results_hist = []
        for _ in range(N_SIMULATIONS):
            # Dla każdej z N_SIMULATIONS prób pozwalamy generatorowi iść dalej,
            # ale proces jest w 100% deterministyczny dzięki nadrzędnemu obiektowi rng
            _, _, _, area_h = run_monte_carlo(n, rng)
            results_hist.append(area_h)

        # Rysowanie rozkładu dla danego N na jednym wykresie
        sns.histplot(
            results_hist,
            bins=HIST_BINS,
            kde=True,
            color=colors[n],
            label=f"N = {n:,}",
            alpha=0.4,
            stat="density",  # Gęstość zamiast surowych zliczeń, aby wykresy były porównywalne
        )

    plt.axvline(
        x=THEORETICAL_INTEGRAL,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Wartość teoretyczna ({THEORETICAL_INTEGRAL})",
    )
    plt.title(
        f"Porównanie rozkładu wyników z {N_SIMULATIONS} symulacji dla różnych wartości N"
    )
    plt.xlabel("Obliczona wartość całki")
    plt.ylabel("Gęstość prawdopodobieństwa (Przeskalowana)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ==========================================
# GŁÓWNY SKRYPT URUCHOMIENIOWY
# ==========================================
def main():
    print("--- ZADANIE 2.B: Monte Carlo ---")

    main_results = {}

    # 1. Przeprowadzenie symulacji dla głównych N
    for n in N_VALUES_MAIN:
        rng_main = np.random.RandomState(SEED)
        x_mc, y_mc, points_under, calka_mc = run_monte_carlo(n, rng_main)
        main_results[n] = (x_mc, y_mc, points_under, calka_mc)
        print(
            f"Całka obliczona (N={n:,}): {calka_mc:.5f} (Teoria: {THEORETICAL_INTEGRAL:.5f})"
        )

    # Generowanie połączonego wykresu rozkładów (Wszystkie N w jednym pliku)
    combined_scatter_file = "monte_carlo_combined.png"
    print(
        f"Generowanie połączonego wykresu rozkładu: {combined_scatter_file}..."
    )
    plot_combined_distributions(main_results, combined_scatter_file)

    # 2. Analiza zbieżności i wykres zbieżności (W osobnym pliku)
    print("Generowanie wykresu zbieżności: Zbieżność Monte Carlo...")
    n_values, mc_results = simulate_convergence(SEED)

    convergence_file = "mc_convergence.png"
    plot_convergence(n_values, mc_results, main_results, convergence_file)

    # 3. Generowanie histogramu przeskalowanego dla różnych N (W osobnym pliku)
    histogram_file = "mc_histogram.png"
    generate_and_plot_histograms(SEED, histogram_file)

    print(
        f"Zapisano wykresy: {combined_scatter_file}, {convergence_file}, {histogram_file}\n"
    )


if __name__ == "__main__":
    main()
