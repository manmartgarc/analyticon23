"""Simple Data Loader for the FIFA data."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import pandas as pd

from analyticon_fifa.directories import DataPath

USECOLS = [
    "sofifa_id",
    "long_name",
    "player_positions",
    "overall",
    "potential",
    "value_eur",
    "wage_eur",
    "age",
    "height_cm",
    "weight_kg",
    "nationality_name",
    "nation_position",
    "nation_jersey_number",
    "preferred_foot",
    "weak_foot",
    "skill_moves",
    "international_reputation",
    "work_rate",
    "body_type",
    "pace",
    "shooting",
    "passing",
    "dribbling",
    "defending",
    "physic",
    "attacking_crossing",
    "attacking_finishing",
    "attacking_heading_accuracy",
    "attacking_short_passing",
    "attacking_volleys",
    "skill_dribbling",
    "skill_curve",
    "skill_fk_accuracy",
    "skill_long_passing",
    "skill_ball_control",
    "movement_acceleration",
    "movement_sprint_speed",
    "movement_agility",
    "movement_reactions",
    "movement_balance",
    "power_shot_power",
    "power_jumping",
    "power_stamina",
    "power_strength",
    "power_long_shots",
    "mentality_aggression",
    "mentality_interceptions",
    "mentality_positioning",
    "mentality_vision",
    "mentality_penalties",
    "mentality_composure",
    "defending_marking_awareness",
    "defending_standing_tackle",
    "defending_sliding_tackle",
    "goalkeeping_diving",
    "goalkeeping_handling",
    "goalkeeping_kicking",
    "goalkeeping_positioning",
    "goalkeeping_reflexes",
    "goalkeeping_speed",
]


class FIFAData(ABC, pd.DataFrame):
    def __new__(cls) -> FIFAData:
        parts: list[pd.DataFrame] = []
        pat = re.compile(r"(\d{2})")
        for path in cls.get_paths():
            year = re.search(pat, path.name)
            if year is None:
                raise ValueError("Couldn't find year in path")
            part = pd.read_csv(path, usecols=USECOLS)
            part["year"] = int(year.group(1)) + 2000
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    @abstractmethod
    def get_paths() -> Iterator[Path]:
        pass


class MaleData(FIFAData):
    @staticmethod
    def get_paths() -> Iterator[Path]:
        return DataPath().get_male_paths()


class FemaleData(FIFAData):
    @staticmethod
    def get_paths() -> Iterator[Path]:
        return DataPath().get_female_paths()


def load_data() -> pd.DataFrame:
    maledf = MaleData()
    femadf = FemaleData()
    maledf["female"] = 0
    femadf["female"] = 1
    df = pd.concat((maledf, femadf), ignore_index=True)
    df = df.sort_values(by=["sofifa_id", "year"])
    df = df.drop_duplicates(subset="sofifa_id", keep="last")
    return df
