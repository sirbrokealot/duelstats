import argparse
from .core.calculations import DSCalc
from .core.plots import DSPlot
from .core.duelstats import DuelStats
import os

def main():
    parser = argparse.ArgumentParser(description="Analyze and plot metrics for deck vs deck games from a CSV file.")
    parser.add_argument("--csv_file", "-c", type=str, help="Path to the CSV file with duel data. Filename is equal to the output dir and file prefix", required=True)
    parser.add_argument("--min_matchup_threshold", type=int, default=3, help="At least X specific deck vs deck duels need to exist before specific matchup is calculated")
    parser.add_argument("--min_evaluation_threshold", type=int, default=3, help="At least X specific matchups have been calculated before global win/loss is calculated.")
    parser.add_argument("--cmap", default='viridis', help='The colormap for the plots',)


    args = parser.parse_args()
    
    csv_file_path = args.csv_file
    filename_without_extension = os.path.splitext(os.path.basename(csv_file_path))[0]
    output_name = filename_without_extension
    os.makedirs(f"results/{output_name}", exist_ok=True)

    duel_stats_instance = DuelStats.load_duel_data(args.csv_file)

    duel_stats_calculations = DSCalc(duel_stats_instance, args.min_matchup_threshold)
    duel_stats_calculations.run()


    plot = DSPlot(duel_stats_instance, cmap=args.cmap, output_name=output_name)
    plot.stats_with_win_loss_summary(min_evaluation_threshold=args.min_evaluation_threshold)
    plot.raw_stats()
    plot.deck_to_deck_duel_count()
    plot.single_deck_duel_count()
    plot.win_loss_vs_cost()

if __name__ == "__main__":
    main()