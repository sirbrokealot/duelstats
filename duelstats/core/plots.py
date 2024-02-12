import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from duelstats.core.duelstats import DuelStats


class DSPlot(DuelStats):
    def __init__(self, duel_stats_instance: DuelStats, cmap: str, output_name: str):
        self.ds = duel_stats_instance
        self.cmap = plt.get_cmap(cmap)
        self.output_name = output_name

    def raw_stats(self,) -> None:
            #todo streamline this with other plots.
            #todo add title with output name
            fig, ax = plt.subplots()
            ax.imshow(self.ds.stats_array, cmap=self.cmap)

            for (i, j), z in np.ndenumerate(self.ds.stats_array):
                ax.text(j, i, "{}".format(z), ha="center", va="center", size=12)

            plt.xticks(np.arange(len(self.ds.deck_names)), self.ds.deck_names)
            plt.yticks(np.arange(len(self.ds.deck_names)), self.ds.deck_names)
            plt.savefig(f"results/{self.output_name}/raw_data_visualization.png")

    def stats_with_win_loss_summary(self, min_evaluation_threshold: int) -> None:

        # Normalize color scale based on actual data range
        valid_data = self.ds.win_loss_data[self.ds.win_loss_data != -1]
        norm = Normalize(vmin=min(valid_data), vmax=max(valid_data))

        fig, ax = plt.subplots()
        cax = ax.imshow(self.ds.win_loss_data, cmap=self.cmap, aspect='auto', interpolation='nearest', norm=norm)

        # Annotate each cell, adjusting for "N/A" and valid data
        for (i, j), value in np.ndenumerate(self.ds.win_loss_data):
            if j == 0 and np.isnan(value):  # Check for NaN in the first column
                text = "N/A"
            else:
                text = f"{value:.1f}"
            ax.text(j, i, text, ha="center", va="center", color="black")


        plt.yticks(np.arange(len(self.ds.deck_names)), self.ds.deck_names)
        xticks_labels = ['Avg W/L%'] + self.ds.deck_names
        plt.xticks(np.arange(len(xticks_labels)), xticks_labels, rotation=45)
        ax.set_title(f"Duel Stats with win loss ratio")
        plt.tight_layout()
        plt.savefig(f"results/{self.output_name}/stats.png")

    def deck_to_deck_duel_count(self) -> None:

        duelcount, deck_names = zip(*((k, val) for k in sorted(self.ds.deck_vs_deck_duel_count) for val in self.ds.deck_vs_deck_duel_count[k]))
        duelcount_list = list(duelcount)
        deck_names_list = [f"{entry[0]} vs {entry[1]}" for entry in deck_names]

        # Map duelcount to color
        norm = plt.Normalize(min(duelcount_list), max(duelcount_list))
        scalar_map = ScalarMappable(norm=norm, cmap=self.cmap)
        colors = [scalar_map.to_rgba(duelcount) for duelcount in duelcount_list]

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(deck_names_list, duelcount_list, color=colors)
        ax.grid(visible=True, color="grey", linestyle="-.", linewidth=0.5, alpha=0.2)

        ax.set_xlabel(f"Deck to Deck duelcount")
        ax.set_ylabel("Deck combination")
        ax.set_title("Find rarest deck vs deck combination...")

        # Explicitly provide the ax argument to plt.colorbar
        plt.colorbar(scalar_map, ax=ax, label='Duel Count')
        fig.tight_layout()

        plt.savefig(f"results/{self.output_name}/deck_to_deck_duelcount.png")


    def single_deck_duel_count(self) -> None:
        single_deck_count = {}
        for deckname in self.ds.deck_names:
            single_deck_count[deckname] = 0
        for count, deck_combinations in sorted(self.ds.deck_vs_deck_duel_count.items()):
            for deck_combination in deck_combinations:
                single_deck_count[deck_combination[0]] += count
                single_deck_count[deck_combination[1]] += count
        
        deck_names_list = list(single_deck_count.keys())
        duelcount_list = list(single_deck_count.values())

        norm = Normalize(vmin=min(duelcount_list), vmax=max(duelcount_list))
        scalar_map = ScalarMappable(norm=norm, cmap=self.cmap)
        colors = scalar_map.to_rgba(duelcount_list)

        fig, ax = plt.subplots(figsize=(10, 8))
        bars = ax.barh(deck_names_list, duelcount_list, color=colors)
        ax.grid(visible=True, color="grey", linestyle="-.", linewidth=0.5, alpha=0.2)

        ax.set_xlabel("duelcount")
        ax.set_ylabel("Single Deck Evaluation")
        ax.set_title("Single deck duelcount")

        cbar = plt.colorbar(scalar_map, ax=ax, label='Duel Count')

        plt.tight_layout()
        plt.savefig(f"results/{self.output_name}/single_deck_duelcount.png")


    def win_loss_vs_cost(self):
        """
        Plots deck win-loss percentages against their financial cost.
        """

        # Convert money_values from string to float for plotting, ensuring alignment with deck_names
        money_values_float = [float(value) for value in self.ds.money_values]

        # Create the scatter plot
        fig, ax = plt.subplots()
        scatter = ax.scatter(money_values_float, self.ds.win_loss_data[:, 0], alpha=0.5)

        # Label each point with the corresponding deck name for identification
        for i, deck_name in enumerate(self.ds.deck_names):
            ax.annotate(deck_name, (money_values_float[i], self.ds.win_loss_data[:, 0][i]))

        ax.set_xlabel('Deck Cost ($)')
        ax.set_ylabel('Average Win-Loss Percentage (%)')
        ax.set_title('Win-Loss Dominance vs. Financial Investment')

        plt.tight_layout()
        plt.savefig(f"results/{self.output_name}/cost_vs_winrate.png")
        plt.show()