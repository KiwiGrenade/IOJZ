import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# ==========================================
# --- STAŁE I KONFIGURACJA ---
# ==========================================
X_MIN = -1.0  # Kwadrat przesunięty, środek w (0,0)
X_MAX = 1.0
Y_MIN = -1.0
Y_MAX = 1.0
THEORETICAL_INTEGRAL = np.pi

SEED = 42
N_VALUES_MAIN = [10000, 100000, 1000000]
N_SIMULATIONS = 100
HIST_BINS = 20
N_CONVERGENCE_SAMPLES = 50
LINE_RESOLUTION = 200  # Zwiększona rozdzielczość dla płynnego okręgu

FIG_SIZE_COMBINED = (18, 5)
FIG_SIZE_CONVERGENCE = (9, 5)
FIG_SIZE_HISTOGRAM = (10, 6)

# ==========================================
# --- LOGIKA MONTE CARLO ---
# ==========================================
def run_monte_carlo(n_points, rng):
    x_mc = rng.uniform(X_MIN, X_MAX, n_points)
    y_mc = rng.uniform(Y_MIN, Y_MAX, n_points)

    # Warunek Pitagorasa dla pełnego koła ze środkiem w (0,0)
    points_inside_circle = (x_mc**2 + y_mc**2) <= 1.0

    # Szacowanie Pi (Stosunek * 4)
    calculated_pi = (np.sum(points_inside_circle) / n_points) * 4.0

    return x_mc, y_mc, points_inside_circle, calculated_pi

def simulate_convergence(seed):
    rng = np.random.RandomState(seed)
    n_values = np.logspace(2, 6, num=N_CONVERGENCE_SAMPLES, dtype=int)
    n_values = np.unique(np.sort(np.concatenate((n_values, N_VALUES_MAIN))))

    mc_results = []
    for n in n_values:
        rng_local = np.random.RandomState(seed)
        _, _, _, pi_estimate = run_monte_carlo(n, rng_local)
        mc_results.append(pi_estimate)

    return n_values, mc_results

# ==========================================
# --- FUNKCJE RYSUJĄCE ---
# ==========================================
def plot_combined_distributions(results_dict, filename):
    fig, axes = plt.subplots(1, len(N_VALUES_MAIN), figsize=FIG_SIZE_COMBINED)

    # Generowanie punktów dla pełnego okręgu (używamy kąta theta od 0 do 2*pi)
    theta = np.linspace(0, 2 * np.pi, LINE_RESOLUTION)
    x_circle = np.cos(theta)
    y_circle = np.sin(theta)

    for idx, n in enumerate(N_VALUES_MAIN):
        ax = axes[idx]
        x_mc, y_mc, points_inside, _ = results_dict[n]

        marker_size = 1.0 if n < 1000000 else 0.1
        alpha_val = 0.6 if n < 1000000 else 0.2

        ax.scatter(
            x_mc[points_inside],
            y_mc[points_inside],
            color="green",
            s=marker_size,
            alpha=alpha_val,
            label="Wewnątrz koła" if idx == 0 else "",
        )
        ax.scatter(
            x_mc[~points_inside],
            y_mc[~points_inside],
            color="red",
            s=marker_size,
            alpha=alpha_val,
            label="Na zewnątrz koła" if idx == 0 else "",
        )

        ax.plot(
            x_circle,
            y_circle,
            color="black",
            linewidth=2,
            label="Krawędź koła" if idx == 0 else "",
        )

        ax.set_title(f"Rozkład punktów dla N = {n:,}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        
        # Wymuszamy równe proporcje osi, aby okrąg nie był spłaszczony (jako elipsa)
        ax.set_aspect('equal', adjustable='box')
        
        ax.set_xlim([X_MIN, X_MAX])
        ax.set_ylim([Y_MIN, Y_MAX])
        ax.grid(True, linestyle=":", alpha=0.5)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    plt.savefig(filename, dpi=200)
    plt.close()

def plot_convergence(n_values, mc_results, main_results, filename):
    plt.figure(figsize=FIG_SIZE_CONVERGENCE)
    plt.plot(n_values, mc_results, label="Ścieżka symulacji", color="blue", alpha=0.7)
    plt.axhline(
        y=THEORETICAL_INTEGRAL, color="red", linestyle="-", linewidth=2,
        label=f"Wartość Pi (Teoria) ≈ {THEORETICAL_INTEGRAL:.5f}"
    )

    for n in N_VALUES_MAIN:
        _, _, _, pi_estimate = main_results[n]
        plt.axvline(x=n, color="gray", linestyle="--", alpha=0.5)
        plt.scatter(
            n, pi_estimate, color="gold", edgecolor="black", s=120, marker="*", zorder=5,
            label=f"Wynik dla N={n:,}: {pi_estimate:.5f}"
        )

    plt.xscale("log")
    plt.title("Zbieżność metody Monte Carlo (Szacowanie liczby Pi)")
    plt.xlabel("Liczba losowanych punktów (N)")
    plt.ylabel("Obliczona wartość Pi")
    plt.legend(loc="lower left")
    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def generate_and_plot_histograms(seed, filename):
    print(f"Uruchamianie {N_SIMULATIONS} symulacji dla każdego N w celu wygenerowania histogramu...")
    plt.figure(figsize=FIG_SIZE_HISTOGRAM)

    colors = {10000: "teal", 100000: "royalblue", 1000000: "purple"}
    rng = np.random.RandomState(seed)

    for n in N_VALUES_MAIN:
        results_hist = []
        for _ in range(N_SIMULATIONS):
            _, _, _, pi_estimate = run_monte_carlo(n, rng)
            results_hist.append(pi_estimate)

        sns.histplot(
            results_hist, bins=HIST_BINS, kde=True, color=colors[n], label=f"N = {n:,}", alpha=0.4, stat="density"
        )

    plt.axvline(
        x=THEORETICAL_INTEGRAL, color="red", linestyle="--", linewidth=2,
        label=f"Wartość teoretyczna (≈{THEORETICAL_INTEGRAL:.5f})"
    )
    plt.title(f"Porównanie rozkładu szacunków Pi z {N_SIMULATIONS} symulacji (różne N)")
    plt.xlabel("Oszacowana wartość liczby Pi")
    plt.ylabel("Gęstość prawdopodobieństwa (Przeskalowana)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# ==========================================
# --- GŁÓWNY SKRYPT ---
# ==========================================
def main():
    print("--- Szacowanie liczby Pi metodą Monte Carlo (Pełne koło) ---")
    main_results = {}

    for n in N_VALUES_MAIN:
        rng_main = np.random.RandomState(SEED)
        x_mc, y_mc, points_inside, pi_estimate = run_monte_carlo(n, rng_main)
        main_results[n] = (x_mc, y_mc, points_inside, pi_estimate)
        print(f"Szacowane Pi (N={n:,}): {pi_estimate:.5f} (Teoria: {THEORETICAL_INTEGRAL:.5f})")

    combined_scatter_file = "pi_monte_carlo_combined.png"
    print(f"Generowanie wykresu rozkładu: {combined_scatter_file}...")
    plot_combined_distributions(main_results, combined_scatter_file)

    print("Generowanie wykresu zbieżności...")
    n_values, mc_results = simulate_convergence(SEED)
    convergence_file = "pi_mc_convergence.png"
    plot_convergence(n_values, mc_results, main_results, convergence_file)

    histogram_file = "pi_mc_histogram.png"
    generate_and_plot_histograms(SEED, histogram_file)

    print(f"\nGotowe! Zapisano wykresy:\n- {combined_scatter_file}\n- {convergence_file}\n- {histogram_file}\n")

if __name__ == "__main__":
    main()
