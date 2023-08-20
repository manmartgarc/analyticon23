from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from analyticon_fifa.data_loader import load_data

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


class FifaPipeline(Pipeline):
    def __new__(cls) -> Pipeline:
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
                        n_estimators=10, n_jobs=-1, random_state=42
                    ),
                ),
            ],
            verbose=True,
        )
        return pipeline


if __name__ == "__main__":
    fp_cv = FifaPipeline()
    df = load_data()
    df = df[df["pace"].notnull()]
    source = df[df["value_eur"].notnull()]
    y = source["value_eur"]
    X = source.drop(columns="value_eur")
    scores = cross_val_score(estimator=fp_cv, X=X, y=y, cv=5, n_jobs=-1)
    print(f"Mean CV R^2: {scores.mean():.2f}")
    fp = FifaPipeline()
    fp.fit(X=X, y=y)
    target = df[df["value_eur"].isnull()]
    # print(f"R2 score: {r2_score(y_test, y_pred):.2f}")
