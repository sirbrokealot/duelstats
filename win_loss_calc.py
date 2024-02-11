from typing import Dict, List
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from matplotlib.colors import Normalize

np.seterr(divide="ignore", invalid="ignore")


CMAP = plt.get_cmap("viridis")

# At least X games need to be played to visualize specific win/loss statistics
MINIMAL_GAMEPLAY_THRESHOLD = 3
# At least X win loss entries need to be in place to visualize global w/l stats.
MINIMAL_WIN_LOSS_THRESHOLD = 3
class GameData:
    def __init__(self, stats_array: np.ndarray, deck_names: list[str], money_values: list[str]):
        self.stats_array = stats_array
        self.deck_names = deck_names
        self.money_values = money_values

    @classmethod
    def load_game_data(cls, file_path: str = "statistics.csv") -> 'GameData':
        """
        Loads game statistics and additional data from a CSV file.

        Args:
            file_path (str): Path to the CSV file containing game data.

        Returns:
            GameData: An object containing the game statistics array, deck names, and money values.
        """

        with open(file_path, 'r') as file:
            lines = file.readlines()


        deck_names = lines[0].strip().split(',')  # first row is deck names
        money_values = lines[1].strip().split(',')  # second row is money values
        stats_data = lines[2:]  # rest of the data is game statistics

        stats_array = np.genfromtxt(stats_data, delimiter=",", dtype=float)
        return cls(stats_array, deck_names, money_values)

    def calculate_total_games_played(self,
        threshold: int = MINIMAL_GAMEPLAY_THRESHOLD,
    ) -> tuple[ndarray, ndarray]:
        """
        Calculates and filters the total games played for the deck pairs.

        Args:
            stats_array: A 2D array of game statistics, where each element [i, j]
                                represents the games played between deck i and deck j.
            threshold: The minimum number of games played required to consider a deck pair.

        Returns:
            - total_games (ndarray): A 2D array with the total games played between each deck pair.
            - total_games_filtered (ndarray): The same as total_games but with entries below the threshold set to NaN.
        """
        total_games = np.empty_like(self.stats_array, dtype=float)
        total_games[:] = np.nan

        # Calculate total games played
        for (i, j), games in np.ndenumerate(self.stats_array):
            if not np.isnan(games):
                total_played = self.stats_array[i, j] + self.stats_array[j, i]
                total_games[i, j] = total_played
                total_games[j, i] = total_played

        # Create a mask for valid entries
        valid_mask = (total_games >= threshold) | np.isnan(total_games)
        total_games_filtered = np.full_like(total_games, np.nan, dtype=float)
        total_games_filtered[valid_mask] = total_games[valid_mask]

        return total_games, total_games_filtered

    def calculate_win_loss_percentages(self, filtered_totals: ndarray) -> ndarray:
        """
        Calculates win-loss for deck pairs based on filtered total games played.
        
        Args:
            game_data (ndarray): A 2D array of game statistics, where each element
                [i, j] represents the games won by deck i against deck j.
            filtered_totals (ndarray): A 2D array of the total games played between
                each deck pair, filtered by a threshold to include only those with
                sufficient play count.
        
        Returns:
            A 2D array where each element [i, j] represents the win percentage of
            deck i against deck j, calculated as (games won / total games played) *
            100. Entries below the threshold are set to NaN.
        """
        win_loss_percentages = self.stats_array / filtered_totals * 100

        return win_loss_percentages


    def calculate_deck_pair_play_count(self, total_games_played: np.ndarray) -> Dict[int, List[List[str]]]:
        """Calculates and groups the deck pair play count.
        
        This function examines the total games played between each pair of decks,
        grouping them by the total count. It's aimed at identifying and suggesting
        the exploration of rarer deck vs. deck combinations for future games.

        Args:
            total_games_played: A symmetric matrix with counts of games played
                                            between each pair of decks.

        Returns:
            Groups of deck pairs by their total play count.
        """
        deck_pair_play_counts: Dict[int, List[List[str]]] = {}
        for i in range(total_games_played.shape[0]):
            for j in range(i + 1, total_games_played.shape[1]):  # Only upper triangle needed since matrix is mirrored
                count = total_games_played[i, j]
                if not np.isnan(count):
                    # Directly append if count exists, else initialize with the current pair
                    pair = [self.deck_names[i], self.deck_names[j]]
                    reverse_pair = [self.deck_names[i], self.deck_names[j]]
                    if count not in deck_pair_play_counts:
                        deck_pair_play_counts[count] = [pair]
                    elif pair not in deck_pair_play_counts[count] and reverse_pair not in deck_pair_play_counts[count]:
                        deck_pair_play_counts[count].append(pair)
        return deck_pair_play_counts


    def plot_stats(self, array_to_plot: np.ndarray, plot_name: str) -> None:

        fig, ax = plt.subplots()
        ax.imshow(array_to_plot, cmap="RdYlGn")

        for (i, j), z in np.ndenumerate(array_to_plot):
            ax.text(j, i, "{}".format(z), ha="center", va="center", size=12)

        plt.xticks(np.arange(len(self.deck_names)), self.deck_names)
        plt.yticks(np.arange(len(self.deck_names)), self.deck_names)
        plt.savefig(f"results/{plot_name}.png")

    def plot_stats_with_win_loss_summary(self, array_to_plot: np.ndarray, plot_name: str) -> None:

        # get mean win loss only for at least three win loss entries.
        non_nan_counts = np.sum(~np.isnan(array_to_plot), axis=1)
        means = np.nanmean(array_to_plot, axis=1, keepdims=True)
        means[non_nan_counts < MINIMAL_WIN_LOSS_THRESHOLD] = -1

        extended_array_to_plot = np.hstack([means, array_to_plot])

        # Normalize color scale based on actual data range
        valid_data = extended_array_to_plot[extended_array_to_plot != -1]
        norm = Normalize(vmin=min(valid_data), vmax=max(valid_data))

        fig, ax = plt.subplots()
        cax = ax.imshow(extended_array_to_plot, cmap="RdYlGn", aspect='auto', interpolation='nearest', norm=norm)

        # Annotate each cell, adjusting for "N/A" and valid data
        for (i, j), value in np.ndenumerate(extended_array_to_plot):
            if value == -1:
                text = "N/A"
                ax.text(j, i, text, ha="center", va="center", color="black")
            else:
                text = f"{value:.1f}"
                ax.text(j, i, text, ha="center", va="center", color="black")

        plt.yticks(np.arange(len(self.deck_names)), self.deck_names)
        xticks_labels = ['Avg W/L%'] + self.deck_names
        plt.xticks(np.arange(len(xticks_labels)), xticks_labels, rotation=45)
        ax.set_title("Game Stats with win loss")
        plt.tight_layout()
        plt.savefig(f"results/{plot_name}.png")

    def plot_deck_combination_counts(self, deck_play_counts: Dict) -> None:
        my_cmap = plt.get_cmap("viridis")

        playcount, deck_names = zip(*((k, val) for k in sorted(deck_play_counts) for val in deck_play_counts[k]))
        playcount_list = list(playcount)
        deck_names_list = [f"{entry[0]} vs {entry[1]}" for entry in deck_names]

        # Map playcount to color
        norm = plt.Normalize(min(playcount_list), max(playcount_list))
        scalar_map = ScalarMappable(norm=norm, cmap=my_cmap)
        colors = [scalar_map.to_rgba(playcount) for playcount in playcount_list]

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(deck_names_list, playcount_list, color=colors)
        ax.grid(visible=True, color="grey", linestyle="-.", linewidth=0.5, alpha=0.2)

        ax.set_xlabel("Playcount")
        ax.set_ylabel("Deck combination")
        ax.set_title("Find rarest deck vs deck combination...")

        # Explicitly provide the ax argument to plt.colorbar
        plt.colorbar(scalar_map, ax=ax, label='Play Count')
        fig.tight_layout()

        plt.savefig("results/combinations_playcount.png")


    def plot_single_deck_counts(self, deck_play_counts: Dict) -> None:
        single_deck_count = {}
        for deckname in self.deck_names:
            single_deck_count[deckname] = 0
        for count, deck_combinations in sorted(deck_play_counts.items()):
            for deck_combination in deck_combinations:
                single_deck_count[deck_combination[0]] += count
                single_deck_count[deck_combination[1]] += count
        
        deck_names_list = list(single_deck_count.keys())
        playcount_list = list(single_deck_count.values())

        my_cmap = plt.get_cmap("viridis")
        norm = Normalize(vmin=min(playcount_list), vmax=max(playcount_list))
        scalar_map = ScalarMappable(norm=norm, cmap=my_cmap)
        colors = scalar_map.to_rgba(playcount_list)

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(deck_names_list, playcount_list, color=colors)
        ax.grid(visible=True, color="grey", linestyle="-.", linewidth=0.5, alpha=0.2)

        ax.set_xlabel("Playcount")
        ax.set_ylabel("Single Deck Evaluation")
        ax.set_title("What deck to play next..")

        cbar = plt.colorbar(scalar_map, ax=ax, label='Play Count')

        plt.tight_layout()
        plt.savefig("results/deck_playcount.png")


    def plot_win_loss_vs_cost(self):
        """
        Plots deck win-loss percentages against their financial cost.
        """
        # Assuming total_games_played_filtered is calculated once and used here directly.
        total_games_played, total_games_played_filtered = self.calculate_total_games_played()
        win_loss_percentages = self.calculate_win_loss_percentages(total_games_played_filtered)

        # Ensure to handle division by zero or undefined cases.
        win_loss_averages = np.nanmean(win_loss_percentages, axis=1)

        # Convert money_values from string to float for plotting, ensuring alignment with deck_names
        money_values_float = [float(value) for value in self.money_values]

        # Create the scatter plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(money_values_float, win_loss_averages, alpha=0.5)

        # Label each point with the corresponding deck name for identification
        for i, deck_name in enumerate(self.deck_names):
            ax.annotate(deck_name, (money_values_float[i], win_loss_averages[i]))

        ax.set_xlabel('Deck Cost ($)')
        ax.set_ylabel('Average Win-Loss Percentage (%)')
        ax.set_title('Win-Loss Dominance vs. Financial Investment')

        plt.tight_layout()
        plt.savefig("results/win_loss_vs_cost.png")
        plt.show()

if __name__ == "__main__":
    gd = GameData.load_game_data()

    total_games_played, total_games_played_filtered  = gd.calculate_total_games_played()
    win_loss_percentages = gd.calculate_win_loss_percentages(total_games_played_filtered)
    deck_play_counts = gd.calculate_deck_pair_play_count(total_games_played)

    gd.plot_deck_combination_counts(deck_play_counts)
    gd.plot_single_deck_counts(deck_play_counts)

    gd.plot_stats_with_win_loss_summary(np.around(win_loss_percentages, decimals=1), "win_loss_game_stats")
    gd.plot_stats(gd.stats_array, "game_stats")
    gd.plot_win_loss_vs_cost()
