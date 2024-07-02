# DuelStats

DuelStats is a Python-based analysis tool designed to analyze and visualize deck
vs. deck game statistics from a CSV file. It calculates win-loss percentages,
duel counts between decks, and visualizes these statistics alongside deck costs
to provide insights into the game's balance and potential "pay-to-win" dynamics.

**TODO's**
- fully overhaul docstrings as they are wrong since the refactoring
- tests
- introduce mypy, ruff
- Adjust the gitignore to not include plots or csv - have the csv example in the readme

## Features

- Load duel game statistics from a CSV file.
- Calculate total duels played between each pair of decks.
- Calculate win-loss percentages for each deck pair.
- Visualize the win-loss statistics, duel counts, and deck costs.
- Support for custom minimum matchup and evaluation thresholds.
- Customizable plots using matplotlib.


## Installation

Ensure you have Python 3.10 and Poetry installed on your system. To install
DuelStats, follow these steps:
```bash
git clone https://github.com/sirbrokealot/duelstats.git
cd duelstats
curl -sSL https://install.python-poetry.org | python3 -
# put poetry to path variable 
i
poetry install
```

## Usage

To run DuelStats with your CSV file containing duel statistics:
```bash
poetry run duelstats -c path/to/your/data.csv
```

## Command Line Arguments

--csv_file or -c: Path to the CSV file with duel data. The filename (without
extension) is used as the output directory and file prefix for generated plots.

--min_matchup_threshold: The minimum number of matchups required to include a
specific deck vs. deck duel in the analysis.

--min_evaluation_threshold: The minimum number of matchups calculated before the
global win/loss statistic is considered valid.

--cmap: Specifies the colormap for the generated plots (default: 'viridis').

## CSV File Format

The CSV file should have the following format
```
DeckName1,DeckName2,DeckName3
5,6,15
,0,0
1,,0
0,0,

```
- First Row (Deck Names): Names of the decks involved in the duel statistics.
- Second Row (Money Values): The cost associated with each deck.
- Subsequent Rows (Duel Data):
    - Each cell represents the win count of the deck. The cell represents how many times the deck from the row won against the deck from the column.
    - The cell representing a fight with identical decks is left empty because a deck doesn't duel against itself. This pattern follows for each deck across the rows.
    - For example, the value of 1 in the second 'duel data' row, first column, signifies that DeckName2 has defeated DeckName1 once.



## Output

DuelStats generates several plots in the results/output_name/ directory,
providing visual insights into the duel statistics, including win-loss
summaries, duel counts, and the relationship between win rates and deck costs.

## Current example decks

Decks are listed https://www.moxfield.com/decks/personal?folder=wOyK2
Priced are taken from an export to tappedout.net then the cheapest tgc player price. Would be nice to have some automatism but the api's for deckmanager sites are non public to my knowledge.
I tried moxfield.com, tappedout.net and deckstats.net


