from __future__ import annotations

from dataclasses import dataclass

from joblib import dump, load
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyticon_fifa.data_loader import DataPath, load_data

FEATURES = {
    "con": [
        "overall",
        "potential",
        "age",
        "height_cm",
        "weight_kg",
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
        "defending_marking_awareness",
        "defending_standing_tackle",
        "defending_sliding_tackle",
        "goalkeeping_diving",
        "goalkeeping_handling",
        "goalkeeping_kicking",
        "goalkeeping_positioning",
        "goalkeeping_reflexes",
    ],
    "cat": [
        "nationality_name",
        "female",
        "nation_position",
        "player_positions",
        "nation_jersey_number",
        "preferred_foot",
        "weak_foot",
        "skill_moves",
        "international_reputation",
        "work_rate",
        "body_type",
    ],
}


@dataclass
class FIFAModel:
    def __post_init__(self) -> None:
        self.pipeline: Pipeline = self._make_pipeline()
        self.data_path = DataPath()
        self.model_name = "model.joblib"

    @classmethod
    def from_joblib(cls) -> FIFAModel:
        cls = FIFAModel()
        cls.pipeline = load(cls.data_path.joinpath(cls.model_name))
        return cls

    def train(self) -> None:
        df = load_data()
        df = df[df["pace"].notnull()]
        source = df[df["value_eur"].notnull()]
        y = source["value_eur"]
        X = source.drop(columns="value_eur")
        self.pipeline.fit(X=X, y=y)

    def write(self) -> None:
        dump(self.pipeline, self.data_path.joinpath(self.model_name))

    def predict(self, X, **predict_params):
        return self.pipeline.predict(X, **predict_params)


    @staticmethod
    def _make_pipeline() -> Pipeline:
        pipeline = Pipeline(
            [
                (
                    "features",
                    ColumnTransformer(
                        [
                            ("scale", StandardScaler(), FEATURES["con"]),
                            (
                                "onehot",
                                OneHotEncoder(
                                    handle_unknown="infrequent_if_exist"
                                ),
                                FEATURES["cat"],
                            ),
                        ]
                    ),
                ),
                (
                    "estimator",
                    RandomForestRegressor(
                        n_estimators=50,
                        max_depth=10,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ],
            verbose=True,
        )
        return pipeline


if __name__ == "__main__":
    model = FIFAModel()
