"""Merges the male and female data together"""

import numpy as np
import pandas as pd

if __name__ == "__main__":
    maledf = pd.read_csv(
        filepath_or_buffer="data/players_22.csv",
        low_memory=False
    )
    femadf = pd.read_csv(
        filepath_or_buffer="data/female_players_22.csv",
        low_memory=False
    )
    df = pd.concat((maledf, femadf), ignore_index=True)
