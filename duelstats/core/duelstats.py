import numpy as np
class DuelStats():
    def __init__(self, stats_array, deck_names, money_values, deck_colours):
        self.stats_array = stats_array
        self.deck_names = deck_names
        self.money_values = money_values
        self.deck_colors = deck_colours
        self.total_duels = None
        self.total_duels_filtered = None
        self.win_loss_data = None
        self.deck_vs_deck_duel_count: dict[int, list[list[str]]] = {}

    @classmethod
    def load_duel_data(cls, file_path: str) -> 'DuelStats':
        """
        Loads duel statistics and additional data from a CSV file.

        Args:
            file_path: Path to the CSV file containing duel data.

        Returns:
            The class object holding initially a statistics array, deck names, and money values.
        """

        with open(file_path, 'r') as file:
            lines = file.readlines()

        deck_names = lines[0].strip().split(',')  # first row is deck names
        deck_colors = lines[1].strip().split(',') # second row represents the decks colours
        money_values = lines[2].strip().split(',')  # second row is money values
        stats_data = lines[3:]  # rest of the data is duel statistics
        stats_array = np.genfromtxt(stats_data, delimiter=",", dtype=float)
        
        return cls(stats_array, deck_names, money_values, deck_colors)