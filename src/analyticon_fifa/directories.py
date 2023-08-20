"""Module that handles the data directory"""
from __future__ import annotations

from pathlib import PosixPath, Path

from typing import Iterator


class DataPath(PosixPath):
    def __new__(cls) -> DataPath:
        """
        Creates a new instance of the DataPath class.
        """
        current = Path(__file__)
        return super().__new__(cls, current.parents[2] / "data")

    def get_female_paths(self) -> Iterator[Path]:
        return self.glob("female*.csv")

    def get_male_paths(self) -> Iterator[Path]:
        return self.glob("players*.csv")
