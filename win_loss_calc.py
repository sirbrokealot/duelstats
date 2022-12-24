from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

np.seterr(divide="ignore", invalid="ignore")

DECK_NAME_ORDER = ["Aristoc.", "Landfall", "Artifact", "Equip", "Sliver", "Spellsl."]
CMAP = plt.get_cmap("viridis")


def load_data() -> np.ndarray:
    filename = "statistics.csv"
    game_statistics_array = np.genfromtxt(filename, delimiter=",", dtype=float)
    return game_statistics_array


def plot_stats(array_to_plot: np.ndarray, plot_name: str) -> None:

    fig, ax = plt.subplots()
    ax.imshow(array_to_plot, cmap="RdYlGn")

    for (i, j), z in np.ndenumerate(array_to_plot):
        ax.text(j, i, "{}".format(z), ha="center", va="center", size=12)

    plt.xticks(np.arange(len(DECK_NAME_ORDER)), DECK_NAME_ORDER)
    plt.yticks(np.arange(len(DECK_NAME_ORDER)), DECK_NAME_ORDER)
    plt.savefig(f"results/{plot_name}.png")


def plot_stats_with_win_loss_summary(array_to_plot: np.ndarray, plot_name: str) -> None:

    # win_loss_summary = np.around(np.nanmean(array_to_plot,axis=1))
    win_loss_summary = np.reshape(np.around(np.nanmean(array_to_plot,axis=1), decimals=1), (-1, array_to_plot.shape[0]))

    extended_array_to_plot = np.hstack([win_loss_summary.T,array_to_plot])

    fig, ax = plt.subplots()
    ax.imshow(extended_array_to_plot, cmap="RdYlGn")

    for (i, j), z in np.ndenumerate(extended_array_to_plot):
        ax.text(j, i, "{}".format(z), ha="center", va="center", size=12)

    plt.yticks(np.arange(len(DECK_NAME_ORDER)), DECK_NAME_ORDER)

    plt.xticks(np.arange(len(DECK_NAME_ORDER)+1), ['w/l']+DECK_NAME_ORDER)
    plt.savefig(f"results/{plot_name}.png")


def figure_and_form_total_games_played_array(
    game_statistics_array: np.ndarray,
    min_count_threshold: int,
) -> np.ndarray:
    total_games_played_array = np.empty(game_statistics_array.shape, dtype=float)
    total_games_played_array[:] = np.nan

    total_games_played_array_filtered = np.empty(game_statistics_array.shape, dtype=float)
    total_games_played_array_filtered[:] = np.nan

    for (i, j), z in np.ndenumerate(game_statistics_array):
        if not np.isnan(z):
            total_games_played = (
                game_statistics_array[i, j] + game_statistics_array[j, i]
            )
            total_games_played_array[i, j] = total_games_played
            total_games_played_array[j, i] = total_games_played
            if total_games_played < min_count_threshold:
                continue
            else:
                total_games_played_array_filtered[i, j] = total_games_played
                total_games_played_array_filtered[j, i] = total_games_played
    return (total_games_played_array, total_games_played_array_filtered)


def figure_and_define_deck_play_counts(total_games_played_array: np.ndarray) -> Dict:
    deck_play_counts: Dict[int, List[List[str]]] = {}
    for (i, j), z in np.ndenumerate(total_games_played_array):
        if not np.isnan(z):
            if z not in deck_play_counts.keys():
                deck_play_counts[int(z)] = [[DECK_NAME_ORDER[i], DECK_NAME_ORDER[j]]]
            else:
                # removes the counterparts since we would end up with [a,b], [b,a]
                if [DECK_NAME_ORDER[j], DECK_NAME_ORDER[i]] in deck_play_counts[int(z)]:
                    continue
                else:
                    deck_play_counts[int(z)].append(
                        [DECK_NAME_ORDER[i], DECK_NAME_ORDER[j]]
                    )
    return deck_play_counts


def plot_deck_combination_counts(deck_play_counts: Dict) -> None:
    my_cmap = plt.get_cmap("viridis")

    playcount, deck_names = zip(
        *((k, val) for k in sorted(deck_play_counts) for val in deck_play_counts[k])
    )

    playcount_list = list(playcount)
    deck_names_list = [f"{entry[0]} vs {entry[1]}" for entry in deck_names]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.2, 0.1, 0.7, 0.8])
    ax.barh(deck_names_list, playcount_list, color=my_cmap.colors)
    ax.grid(visible=True, color="grey", linestyle="-.", linewidth=0.5, alpha=0.2)

    ax.set_xlabel("Playcount")
    ax.set_ylabel("Deck combination")
    ax.set_title("What combination to play next..")
    plt.savefig("results/combination_play_statistics.png")


def plot_single_deck_counts(deck_play_counts: Dict) -> None:
    single_deck_count = {}
    for deckname in DECK_NAME_ORDER:
        single_deck_count[deckname] = 0
    for i, j in sorted(deck_play_counts.items()):
        for deck_combination in j:
            single_deck_count[deck_combination[0]] += i
            single_deck_count[deck_combination[1]] += i
    playcount_list = list(single_deck_count.values())
    deck_names_list = list(single_deck_count.keys())

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_axes([0.2, 0.1, 0.7, 0.8])
    ax.barh(deck_names_list, playcount_list, color=CMAP.colors)
    ax.grid(visible=True, color="grey", linestyle="-.", linewidth=0.5, alpha=0.2)

    ax.set_xlabel("Playcount")
    ax.set_ylabel("Single Deck Evaluation")
    ax.set_title("What deck to play next..")
    plt.savefig("results/deck_play_statistics.png")


if __name__ == "__main__":
    game_statistics_array = load_data()

    total_games_played_array, total_games_played_array_filtered  = figure_and_form_total_games_played_array(
        game_statistics_array,
        min_count_threshold=3
    )
    win_loss_percentages_array = game_statistics_array / total_games_played_array_filtered * 100
    deck_play_counts = figure_and_define_deck_play_counts(total_games_played_array)

    plot_deck_combination_counts(deck_play_counts)
    plot_single_deck_counts(deck_play_counts)
    plot_stats_with_win_loss_summary(np.around(win_loss_percentages_array, decimals=1), "win_loss_ratio_sum")
    # plot_stats(np.around(win_loss_percentages_array, decimals=1), "win_loss_ratio")

    plot_stats(game_statistics_array, "game_stats")
