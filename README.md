# Size the Model to the Cloudy Spell, Not the Average Sun

Code and data for the paper of that title (IEEE AIoT 2026, workshop Sustainable AIoT for AfroAsian Communities,
paper 244; Varun Puttagunta and Joshua Anojulu).

Everything in the paper is computed by the files here. No hardware was used: the energies and recalls of the
three models are illustrative values, as the paper says.

## What is here

| path | content |
|---|---|
| `code/solar_node.py` | the node model, the policies, exact evaluation, relative value iteration, replay of recorded days |
| `code/experiments.py` | every computation of the paper; each sub-command writes one file under `results/` |
| `code/make_numbers.py` | turns `results/` into the numbers and tables of the paper and checks every claim of the text |
| `code/make_figures.py` | the two figures |
| `code/verify_digits.py` | checks a data file against its checksums |
| `data/` | daily solar irradiance 2021 to 2025 at Dhaka, Ibadan and Niamey (NASA POWER), with hashes; see `data/README_data.md` |
| `results/` | the result files behind the paper, `numbers_catalogue.tsv` (every number of the paper with its source) and `claims_checked.txt` |

## Run

Python 3.10 or later with `numpy`, `scipy` and `matplotlib`.

```
cd code
python3 experiments.py grid 1,2,5,10   # about 2 minutes
python3 experiments.py grid 20         # about 15 minutes
python3 experiments.py night 1,2,5,10
python3 experiments.py night 20
python3 experiments.py checks
python3 experiments.py robust
python3 experiments.py sites           # about 10 minutes
python3 experiments.py menu            # about 20 minutes
python3 make_numbers.py                # numbers and tables of the paper (LaTeX), written to paper/
python3 make_figures.py                # figures, written to paper/figs/ (needs LaTeX for the labels)
```

All runs are deterministic (the Monte Carlo check and the shuffles use fixed seeds).

## The model in one paragraph

Hourly steps. The battery holds up to C joules in units of 6 mJ. A day is sunny or cloudy and keeps its weather with
probability p at midnight. A sunny day harvests H joules along a half-sine from 06:00 to 18:00, a cloudy day 0.2 H.
Every hour costs a sleep floor of 36 mJ (10 microwatts); an hour in which a model runs costs 60 inferences on top
(S 0.5 mJ, M 2 mJ, L 8 mJ per inference; recall 0.80, 0.90, 0.95). If battery plus the hour's harvest cannot pay for
the hour the node browns out and detects nothing in that hour. Charge above C spills. The detection rate is the
long-run share of events that fall in an hour in which a model runs and are caught by it.

## Data

NASA POWER Daily API v2.10.5, parameter ALLSKY_SFC_SW_DWN (source CERES SYN1deg), retrieved 25 November 2026.
The data were obtained from the NASA Langley Research Center POWER project, funded through the NASA Earth Science
Division.

## Use of AI tools

The code, the analysis and the text of the paper were prepared with an LLM-based assistant (Claude, Anthropic) under
the authors' direction, as the paper discloses.
