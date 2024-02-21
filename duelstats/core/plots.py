import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
from duelstats.core.duelstats import DuelStats
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
from matplotlib.patches import Patch

from matplotlib.font_manager import FontProperties

MANA_COLORS = {
    'G': '#a3c095',  # Forest
    'U': '#c1d7e9',  # Island
    'W': '#f8f6d8',  # Plains
    'R': '#e49977',  # Mountain
    'B': '#cac5c0'   # Swamp
}

CUSTOM_FONT_PATH = 'duelstats/fonts/MagicSymbols2008.ttf'
class DSPlot(DuelStats):
    def __init__(self, duel_stats_instance: DuelStats, cmap: str, output_name: str):
        self.ds = duel_stats_instance
        self.cmap = plt.get_cmap(cmap)
        self.output_name = output_name
        self.mana_font = FontProperties(fname=CUSTOM_FONT_PATH, size=12)
        self.mana_symbol_mapping = {deck_name: deck_color for deck_name, deck_color in zip(self.ds.deck_names, self.ds.deck_colors)}

    def raw_stats(self) -> None:
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(self.ds.stats_array, cmap=self.cmap)
        ax.set_title(f"{self.output_name} - Raw Stats Visualization")

        for (i, j), z in np.ndenumerate(self.ds.stats_array):
            ax.text(j, i, f"{z}", ha="center", va="center", size=12)

        plt.xticks(np.arange(len(self.ds.deck_names)), self.ds.deck_names, rotation=45, ha="right")
        plt.yticks(np.arange(len(self.ds.deck_names)), self.ds.deck_names)
        self.add_mtg_symbols_to_ticks(ax, end_offset_x=-0.04, end_offset_y=-0.3, 
                              space_offset_x=-0.031, alignment_offset_x=0.016)

        colorbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.set_ylabel('Wincount - Row vs Column', rotation=90, labelpad=10)

        plt.tight_layout()
        plt.savefig(f"results/{self.output_name}/raw_data_visualization.png", dpi=300)

    def stats_with_win_loss_summary(self) -> None:

        # Normalize color scale based on actual data range
        valid_data = self.ds.win_loss_data[self.ds.win_loss_data != -1]
        norm = Normalize(vmin=min(valid_data), vmax=max(valid_data))

        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.imshow(self.ds.win_loss_data, cmap=self.cmap)
        ax.set_title(f"{self.output_name} - Duel Stats With Win/Loss Ratio")

        # Annotate each cell, adjusting for "N/A" and valid data
        for (i, j), value in np.ndenumerate(self.ds.win_loss_data):
            if j == 0 and np.isnan(value):  # Check for NaN in the first column
                text = "N/A"
            else:
                text = f"{value:.1f}"
            ax.text(j, i, text, ha="center", va="center", color="black", size=12)

        xticks_labels = ['Avg W/L%'] + self.ds.deck_names
        plt.xticks(np.arange(len(xticks_labels)), xticks_labels, rotation=45)
        plt.yticks(np.arange(len(self.ds.deck_names)), self.ds.deck_names)

        self.add_mtg_symbols_to_ticks(ax, end_offset_x=-0.04, end_offset_y=-0.3, 
                              space_offset_x=-0.028, alignment_offset_x=0.014)
        
        colorbar = plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        colorbar.ax.set_ylabel('Average Win/Loss (%) and Win/Loss (%) Row vs Column', rotation=90, labelpad=10)
        
        plt.tight_layout()
        plt.savefig(f"results/{self.output_name}/stats.png", dpi=300)

    def deck_to_deck_duel_count(self) -> None:

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel("Deck to Deck Duelcount")
        ax.set_ylabel("Deck combination")
        ax.set_title(f"{self.output_name} - Find Rarest Deck Vs Deck Combination...")

        duelcount, deck_names = zip(*((k, val) for k in sorted(self.ds.deck_vs_deck_duel_count) for val in self.ds.deck_vs_deck_duel_count[k]))
        duelcount_list = list(duelcount)
        deck_names_list = [f"{entry[0]} vs {entry[1]}" for entry in deck_names]


        norm = plt.Normalize(min(duelcount_list), max(duelcount_list))
        scalar_map = ScalarMappable(norm=norm, cmap=self.cmap)
        colors = [scalar_map.to_rgba(duelcount) for duelcount in duelcount_list]

        bars = ax.barh(deck_names_list, duelcount_list, color=colors)

        plt.colorbar(scalar_map, ax=ax, label=' Deck to Deck Duelcount')
        fig.tight_layout()

        plt.savefig(f"results/{self.output_name}/deck_to_deck_duelcount.png")


    def single_deck_duel_count(self) -> None:

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_xlabel("Duelcount")
        ax.set_title(f"{self.output_name} - Single Deck Duelcount")

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

        ax.barh(deck_names_list, duelcount_list, color=colors)
        plt.colorbar(scalar_map, ax=ax, label='Duel Count')

        self.add_mtg_symbols_to_ticks(ax, end_offset_x=-0.04, end_offset_y=0.3, 
                              space_offset_x=-0.028, alignment_offset_x=0.014)

        fig.tight_layout()
        plt.savefig(f"results/{self.output_name}/single_deck_duelcount.png")


    def win_loss_vs_cost(self):
        """
        Plots deck win-loss percentages against their financial cost.
        """
        money_values_float = [float(value) for value in self.ds.money_values]

        fig, ax = plt.subplots()
        scatter = ax.scatter(money_values_float, self.ds.win_loss_data[:, 0], alpha=0.5)

        for i, deck_name in enumerate(self.ds.deck_names):
            ax.annotate(deck_name, (money_values_float[i], self.ds.win_loss_data[:, 0][i]))

        ax.set_xlabel('Deck Cost ($)')
        ax.set_ylabel('Average Win-Loss Percentage (%)')
        ax.set_title('Win-Loss Dominance vs. Financial Investment')

        plt.tight_layout()
        plt.savefig(f"results/{self.output_name}/cost_vs_winrate.png")
        plt.show()

    def mana_diversity(self):
        color_count = {'G': 0, 'U': 0, 'W': 0, 'R': 0, 'B': 0}
        for colors in self.ds.deck_colors:
            for color in colors:
                if color in color_count:
                    color_count[color] += 1

        labels = color_count.keys()
        sizes = color_count.values()
        colors = [MANA_COLORS[label] for label in labels]

        # Calculate total color usages for display in legend
        total_counts = sum(color_count.values())

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.pie(sizes, labels=[f"{label} ({count})" for label, count in color_count.items()], 
            colors=colors, autopct=lambda pct: f"{pct:.1f}%", startangle=140)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        ax.set_title(f"{self.output_name} - Mana Color Diversity")

        # Custom legend with color counts
        legend_labels = [f"{label}: {count} times" for label, count in color_count.items()]
        patches = [Patch(color=colors[i], label=legend_labels[i]) for i, label in enumerate(labels)]
        plt.legend(handles=patches, title="Mana Colors Usage")

        plt.savefig(f"results/{self.output_name}/mana_diversity.png", dpi=300)
 



    def add_mtg_symbols_to_ticks(self, ax, end_offset_x, end_offset_y, 
                              space_offset_x, alignment_offset_x):
        for i, tick_pos in enumerate(ax.get_yticks()):
            """Adds mtg mana based symbols to decks to better visually define their realm.

            Mathplotlib does not support svg or an easy way of adding imagery to the ticks.
            This function adds some additional ticks with a different font.
            Fiddeling is required to align the symbols correctly. For all your fiddeling needs
            multiple offset parameters are present and explained below.
            
            Args:
                ax: the plots axes information
                end_offset_x: End offset of the additional ticks in x-direction
                end_offset_y: End offset of the additional ticks in y-direction
                space_offset_x: Space offset if multiple symbols are chained in x-direction
                alignment_offset_x: Alignment offset that allows for custom spacing between chars
                    Especially usefull is font's are used that follow the idea of overlaying
                    multiple characters on top of each other to create the final symbol.
            """
            if i >= len(self.ds.deck_names):
                break

            deck_name = self.ds.deck_names[i]
            mana_symbols = self.mana_symbol_mapping.get(deck_name, "")
            for j, mana_symbol in enumerate(mana_symbols):
                symbol_y = tick_pos + end_offset_y
                symbol_x = end_offset_x + (j * space_offset_x)

                color = MANA_COLORS.get(mana_symbol.upper(), 'black')
                transform = ax.get_yaxis_transform()

                # Draw the background circle for the mana symbol
                ax.text(symbol_x, symbol_y, 'O', ha='center', va='center', color=color,
                        fontproperties=self.mana_font, fontsize=14, transform=transform)

                # Draw the mana black mana symbol on top of circle
                ax.text(symbol_x + alignment_offset_x, symbol_y, mana_symbol.upper(), 
                        ha='center', va='center', color='black',
                        fontproperties=self.mana_font, fontsize=14, transform=transform)