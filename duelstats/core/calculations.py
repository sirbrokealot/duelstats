

import numpy as np
from duelstats.core.duelstats import DuelStats


class DSCalc(DuelStats):
    def __init__(self, duel_stats_instance, min_matchup_threshold: int):
        self.ds = duel_stats_instance
        self.min_matchup_threshold = min_matchup_threshold

    def run(self):
        self.total_played_duels()
        self.win_loss()
        self.deck_to_deck_duel_count()

    def total_played_duels(self):
        """
        Calculates and filters the total duels played for the deck pairs.

        Args:
            stats_array: A 2D array of duel statistics, where each element [i, j]
                                represents the duels played between deck i and deck j.

        Returns:
            - total_duels (ndarray): A 2D array with the total duels played between each deck pair.
            - total_duels_filtered (ndarray): The same as total_duels but with entries below the threshold set to NaN.
        """
        self.ds.total_duels = np.empty_like(self.ds.stats_array, dtype=float)
        self.ds.total_duels[:] = np.nan

        # Calculate total duels played
        for (i, j), duels in np.ndenumerate(self.ds.stats_array):
            if not np.isnan(duels):
                agregated_count = self.ds.stats_array[i, j] + self.ds.stats_array[j, i]
                self.ds.total_duels[i, j] = agregated_count
                self.ds.total_duels[j, i] = agregated_count

        # Create a mask for valid entries
        valid_mask = (self.ds.total_duels >= self.min_matchup_threshold) | np.isnan(self.ds.total_duels)
        self.ds.total_duels_filtered = np.full_like(self.ds.total_duels, np.nan, dtype=float)
        self.ds.total_duels_filtered[valid_mask] = self.ds.total_duels[valid_mask]

    def win_loss(self):
        """
        Calculates win-loss for deck pairs based on filtered total duels played.
        
        Args:
            duel_data (ndarray): A 2D array of duel statistics, where each element
                [i, j] represents the duels won by deck i against deck j.
            filtered_totals (ndarray): A 2D array of the total duels played between
                each deck pair, filtered by a threshold to include only those with
                sufficient play count.
        
        Returns:
            A 2D array where each element [i, j] represents the win percentage of
            deck i against deck j, calculated as (duels won / total duels played) *
            100. Entries below the threshold are set to NaN.
        """
        win_loss_percentages = self.ds.stats_array / self.ds.total_duels_filtered * 100

        # Shape (n, 1) for n decks.
        win_loss_means = np.zeros((win_loss_percentages.shape[0], 1))

        for idx in range(win_loss_percentages.shape[0]):
            row = win_loss_percentages[idx, :]
            # Check if the row is not entirely NaN to avoid RuntimeWarning.
            if not np.all(np.isnan(row)):
                win_loss_means[idx] = np.nanmean(row)
        else:
            win_loss_means[idx] = np.nan

        self.ds.win_loss_data = np.hstack([win_loss_means, win_loss_percentages])


    def deck_to_deck_duel_count(self):
        """Calculates and groups the deck pair play count.
        
        This function examines the total duels played between each pair of decks,
        grouping them by the total count. It's aimed at identifying and suggesting
        the exploration of rarer deck vs. deck combinations for future duels.

        Args:
            total_duels_played: A symmetric matrix with counts of duels played
                                            between each pair of decks.

        Returns:
            Groups of deck pairs by their total play count.
        """

        for i in range(self.ds.total_duels.shape[0]):
            # Only upper triangle needed since matrix is mirrored
            for j in range(i + 1, self.ds.total_duels.shape[1]):  
                count = self.ds.total_duels[i, j]
                if not np.isnan(count):
                    # Directly append if count exists, else initialize with the current pair
                    pair = [self.ds.deck_names[i], self.ds.deck_names[j]]
                    reverse_pair = [self.ds.deck_names[j], self.ds.deck_names[i]]
                    if count not in self.ds.deck_vs_deck_duel_count:
                        self.ds.deck_vs_deck_duel_count[count] = [pair]
                    elif pair not in self.ds.deck_vs_deck_duel_count[count] and reverse_pair not in self.ds.deck_vs_deck_duel_count[count]:
                        self.ds.deck_vs_deck_duel_count[count].append(pair)



