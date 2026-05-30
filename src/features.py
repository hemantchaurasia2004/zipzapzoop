"""Feature engineering for traffic demand prediction."""
from __future__ import annotations

import json
from pathlib import Path

import geohash
import numpy as np
import pandas as pd

from .config import SMOOTHING_M, SLOTS_PER_DAY


def _decode_geohash(g: str) -> tuple[float, float]:
    try:
        return geohash.decode(g)
    except Exception:
        return np.nan, np.nan


def _neighbors(g: str) -> list[str]:
    try:
        return list(geohash.neighbors(g))
    except Exception:
        return []


def _smooth_mean(count: pd.Series, mean: pd.Series, global_mean: float, m: float) -> pd.Series:
    return (count * mean + m * global_mean) / (count + m)


class FeatureEngineer:
    """Fit on training rows only; transform train or test without target leakage."""

    def __init__(self, smoothing_m: float = SMOOTHING_M):
        self.smoothing_m = smoothing_m
        self.global_demand_mean_: float = 0.0
        self.global_demand_std_: float = 0.0
        self.impute_maps_: dict = {}
        self.encoding_maps_: dict = {}
        self.day48_lookup_: pd.DataFrame | None = None
        self.day49_prior_lookup_: pd.DataFrame | None = None
        self.geohash_neighbor_lookup_: pd.DataFrame | None = None
        self.grid_bounds_: dict = {}
        self.fitted_ = False

    def fit(self, train_df: pd.DataFrame) -> "FeatureEngineer":
        df = train_df.copy()
        if "demand" not in df.columns:
            raise ValueError("fit() requires demand column")

        self.global_demand_mean_ = float(df["demand"].mean())
        self.global_demand_std_ = float(df["demand"].std())

        # Imputation maps
        self.impute_maps_ = {
            "road_by_geohash": df.groupby("geohash")["RoadType"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else "Residential")
            .to_dict(),
            "temp_by_geohash_hour": df.groupby(["geohash", "hour"])["Temperature"].median().to_dict(),
            "temp_by_geohash": df.groupby("geohash")["Temperature"].median().to_dict(),
            "weather_by_geohash_hour": df.groupby(["geohash", "hour"])["Weather"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else "Sunny")
            .to_dict(),
            "weather_by_geohash": df.groupby("geohash")["Weather"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) else "Sunny")
            .to_dict(),
            "road_global": df["RoadType"].mode().iloc[0] if len(df["RoadType"].mode()) else "Residential",
            "weather_global": df["Weather"].mode().iloc[0] if len(df["Weather"].mode()) else "Sunny",
            "temp_global": float(df["Temperature"].median()),
        }

        # Target encodings (Bayesian smoothed)
        g = df.groupby("geohash")["demand"].agg(["mean", "std", "count"]).reset_index()
        g["enc_geohash_mean"] = _smooth_mean(
            g["count"], g["mean"], self.global_demand_mean_, self.smoothing_m
        )
        g["enc_geohash_std"] = g["std"].fillna(self.global_demand_std_)

        gh = df.groupby(["geohash", "hour"])["demand"].agg(["mean", "count"]).reset_index()
        gh["enc_geohash_hour_mean"] = _smooth_mean(
            gh["count"], gh["mean"], self.global_demand_mean_, self.smoothing_m
        )

        rt = df.groupby("RoadType")["demand"].agg(["mean", "count"]).reset_index()
        rt["enc_roadtype_mean"] = _smooth_mean(
            rt["count"], rt["mean"], self.global_demand_mean_, self.smoothing_m
        )

        wx = df.groupby("Weather")["demand"].agg(["mean", "count"]).reset_index()
        wx["enc_weather_mean"] = _smooth_mean(
            wx["count"], wx["mean"], self.global_demand_mean_, self.smoothing_m
        )

        self.encoding_maps_ = {
            "geohash": g.set_index("geohash")[["enc_geohash_mean", "enc_geohash_std"]].to_dict("index"),
            "geohash_hour": {
                f"{row.geohash}|{int(row.hour)}": row.enc_geohash_hour_mean
                for row in gh.itertuples()
            },
            "roadtype": rt.set_index("RoadType")["enc_roadtype_mean"].to_dict(),
            "weather": wx.set_index("Weather")["enc_weather_mean"].to_dict(),
        }

        # Day 48 same-slot demand (strongest lag)
        d48 = df[df["day"] == 48][["geohash", "timestamp", "demand"]].rename(
            columns={"demand": "demand_day48_same_slot"}
        )
        self.day48_lookup_ = d48.drop_duplicates(subset=["geohash", "timestamp"])

        # Day 49 prior slots within training data
        d49 = df[df["day"] == 49].sort_values(["geohash", "slot_idx"])
        d49 = d49.assign(
            demand_lag_1slot=d49.groupby("geohash")["demand"].shift(1),
            demand_lag_4slot=d49.groupby("geohash")["demand"].shift(4),
            demand_roll4_mean=d49.groupby("geohash")["demand"]
            .transform(lambda s: s.shift(1).rolling(4, min_periods=1).mean()),
        )
        self.day49_prior_lookup_ = d49[
            ["geohash", "timestamp", "demand_lag_1slot", "demand_lag_4slot", "demand_roll4_mean"]
        ].drop_duplicates(subset=["geohash", "timestamp"])

        # Neighbor demand on day 48 per timestamp
        geo_neighbors = {g: _neighbors(g) for g in df["geohash"].unique()}
        rows = []
        for ts, grp in d48.groupby("timestamp"):
            slot_map = grp.set_index("geohash")["demand_day48_same_slot"].to_dict()
            for g, _ in slot_map.items():
                nbs = geo_neighbors.get(g, [])
                vals = [slot_map[n] for n in nbs if n in slot_map]
                rows.append(
                    {
                        "geohash": g,
                        "timestamp": ts,
                        "neighbor_d48_mean": np.mean(vals) if vals else np.nan,
                        "neighbor_d48_median": np.median(vals) if vals else np.nan,
                    }
                )
        self.geohash_neighbor_lookup_ = pd.DataFrame(rows)

        # Grid normalization
        lats, lons = zip(*[_decode_geohash(g) for g in df["geohash"].unique()])
        lats, lons = np.array(lats), np.array(lons)
        valid = ~np.isnan(lats)
        self.grid_bounds_ = {
            "lat_min": float(np.nanmin(lats[valid])),
            "lat_max": float(np.nanmax(lats[valid])),
            "lon_min": float(np.nanmin(lons[valid])),
            "lon_max": float(np.nanmax(lons[valid])),
        }

        self.fitted_ = True
        return self

    def _vectorized_impute(self, out: pd.DataFrame) -> pd.DataFrame:
        road_map = self.impute_maps_["road_by_geohash"]
        out["RoadType"] = out["RoadType"].replace("", np.nan)
        out["RoadType"] = out["RoadType"].fillna(out["geohash"].map(road_map)).fillna(
            self.impute_maps_["road_global"]
        )

        temp_gh = pd.Series(self.impute_maps_["temp_by_geohash_hour"])
        if len(temp_gh):
            temp_gh.index = pd.MultiIndex.from_tuples(temp_gh.index, names=["geohash", "hour"])
            out = out.merge(
                temp_gh.rename("temp_gh").reset_index(),
                on=["geohash", "hour"],
                how="left",
            )
            out["Temperature"] = out["Temperature"].fillna(out["temp_gh"])
            out = out.drop(columns=["temp_gh"])
        out["Temperature"] = out["Temperature"].fillna(
            out["geohash"].map(self.impute_maps_["temp_by_geohash"])
        ).fillna(self.impute_maps_["temp_global"])

        out["Weather"] = out["Weather"].replace("", np.nan)
        w_gh = pd.Series(self.impute_maps_["weather_by_geohash_hour"])
        if len(w_gh):
            w_gh.index = pd.MultiIndex.from_tuples(w_gh.index, names=["geohash", "hour"])
            out = out.merge(
                w_gh.rename("weather_gh").reset_index(),
                on=["geohash", "hour"],
                how="left",
            )
            out["Weather"] = out["Weather"].fillna(out["weather_gh"])
            out = out.drop(columns=["weather_gh"])
        out["Weather"] = out["Weather"].fillna(
            out["geohash"].map(self.impute_maps_["weather_by_geohash"])
        ).fillna(self.impute_maps_["weather_global"])
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Call fit() before transform()")

        out = df.copy()
        out = self._vectorized_impute(out)

        # Geohash spatial
        coords = out["geohash"].map(lambda g: _decode_geohash(g))
        out["lat"] = coords.map(lambda x: x[0])
        out["lon"] = coords.map(lambda x: x[1])
        out["geohash_prefix2"] = out["geohash"].str[:2]
        out["geohash_prefix4"] = out["geohash"].str[:4]

        b = self.grid_bounds_
        lat_rng = b["lat_max"] - b["lat_min"] + 1e-9
        lon_rng = b["lon_max"] - b["lon_min"] + 1e-9
        out["grid_x"] = (out["lon"] - b["lon_min"]) / lon_rng
        out["grid_y"] = (out["lat"] - b["lat_min"]) / lat_rng

        # Cyclical time
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
        out["slot_sin"] = np.sin(2 * np.pi * out["slot_idx"] / SLOTS_PER_DAY)
        out["slot_cos"] = np.cos(2 * np.pi * out["slot_idx"] / SLOTS_PER_DAY)

        # Interactions
        out["weather_hour"] = out["Weather"].astype(str) + "_" + out["hour"].astype(str)
        out["road_lanes"] = out["RoadType"].astype(str) + "_" + out["NumberofLanes"].astype(str)
        weather_codes = pd.Categorical(out["Weather"]).codes.astype(float)
        out["temp_x_weather"] = out["Temperature"] * weather_codes

        # Target encodings
        out["enc_geohash_mean"] = out["geohash"].map(
            lambda g: self.encoding_maps_["geohash"].get(g, {}).get(
                "enc_geohash_mean", self.global_demand_mean_
            )
        )
        out["enc_geohash_std"] = out["geohash"].map(
            lambda g: self.encoding_maps_["geohash"].get(g, {}).get(
                "enc_geohash_std", self.global_demand_std_
            )
        )
        out["enc_geohash_hour_mean"] = [
            self.encoding_maps_["geohash_hour"].get(f"{g}|{int(h)}", self.global_demand_mean_)
            for g, h in zip(out["geohash"], out["hour"])
        ]
        out["enc_roadtype_mean"] = out["RoadType"].map(
            lambda r: self.encoding_maps_["roadtype"].get(r, self.global_demand_mean_)
        )
        out["enc_weather_mean"] = out["Weather"].map(
            lambda w: self.encoding_maps_["weather"].get(w, self.global_demand_mean_)
        )

        # Lags from day 48
        out = out.merge(self.day48_lookup_, on=["geohash", "timestamp"], how="left")
        out = out.merge(self.day49_prior_lookup_, on=["geohash", "timestamp"], how="left")
        out = out.merge(self.geohash_neighbor_lookup_, on=["geohash", "timestamp"], how="left")

        # Fill lag fallbacks
        out["demand_day48_same_slot"] = out["demand_day48_same_slot"].fillna(out["enc_geohash_hour_mean"])
        out["demand_lag_1slot"] = out["demand_lag_1slot"].fillna(out["demand_day48_same_slot"])
        out["demand_lag_4slot"] = out["demand_lag_4slot"].fillna(out["demand_day48_same_slot"])
        out["demand_roll4_mean"] = out["demand_roll4_mean"].fillna(out["enc_geohash_mean"])
        out["neighbor_d48_mean"] = out["neighbor_d48_mean"].fillna(out["enc_geohash_mean"])
        out["neighbor_d48_median"] = out["neighbor_d48_median"].fillna(out["enc_geohash_mean"])

        # LargeVehicles / Landmarks as numeric flags
        out["large_vehicles_allowed"] = (out["LargeVehicles"] == "Allowed").astype(int)
        out["has_landmarks"] = (out["Landmarks"] == "Yes").astype(int)

        return out

    def get_feature_columns(self) -> list[str]:
        return [
            "lat",
            "lon",
            "grid_x",
            "grid_y",
            "hour",
            "minute",
            "slot_idx",
            "day",
            "hour_sin",
            "hour_cos",
            "slot_sin",
            "slot_cos",
            "NumberofLanes",
            "Temperature",
            "temp_x_weather",
            "large_vehicles_allowed",
            "has_landmarks",
            "demand_day48_same_slot",
            "demand_lag_1slot",
            "demand_lag_4slot",
            "demand_roll4_mean",
            "neighbor_d48_mean",
            "neighbor_d48_median",
            "enc_geohash_mean",
            "enc_geohash_std",
            "enc_geohash_hour_mean",
            "enc_roadtype_mean",
            "enc_weather_mean",
            "geohash",
            "RoadType",
            "Weather",
            "LargeVehicles",
            "Landmarks",
            "geohash_prefix2",
            "geohash_prefix4",
            "weather_hour",
            "road_lanes",
        ]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "smoothing_m": self.smoothing_m,
            "global_demand_mean": self.global_demand_mean_,
            "global_demand_std": self.global_demand_std_,
            "impute_maps": self.impute_maps_,
            "encoding_maps": self.encoding_maps_,
            "grid_bounds": self.grid_bounds_,
        }
        # JSON-safe impute keys (tuples -> strings)
        impute = meta["impute_maps"].copy()
        for key in ("temp_by_geohash_hour", "weather_by_geohash_hour"):
            impute[key] = {f"{g}|{h}": v for (g, h), v in impute[key].items()}
        meta["impute_maps"] = impute
        (path / "meta.json").write_text(json.dumps(meta, default=str))
        self.day48_lookup_.to_parquet(path / "day48_lookup.parquet", index=False)
        self.day49_prior_lookup_.to_parquet(path / "day49_prior_lookup.parquet", index=False)
        self.geohash_neighbor_lookup_.to_parquet(path / "neighbor_lookup.parquet", index=False)

    @classmethod
    def load(cls, path: Path) -> "FeatureEngineer":
        path = Path(path)
        fe = cls()
        meta = json.loads((path / "meta.json").read_text())
        fe.smoothing_m = meta["smoothing_m"]
        fe.global_demand_mean_ = meta["global_demand_mean"]
        fe.global_demand_std_ = meta["global_demand_std"]
        impute = meta["impute_maps"]
        for key in ("temp_by_geohash_hour", "weather_by_geohash_hour"):
            impute[key] = {
                (k.split("|")[0], int(k.split("|")[1])): v for k, v in impute[key].items()
            }
        fe.impute_maps_ = impute
        fe.encoding_maps_ = meta["encoding_maps"]
        fe.grid_bounds_ = meta["grid_bounds"]
        fe.day48_lookup_ = pd.read_parquet(path / "day48_lookup.parquet")
        fe.day49_prior_lookup_ = pd.read_parquet(path / "day49_prior_lookup.parquet")
        fe.geohash_neighbor_lookup_ = pd.read_parquet(path / "neighbor_lookup.parquet")
        fe.fitted_ = True
        return fe


def build_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    fit_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, FeatureEngineer]:
    """Fit feature engineer on fit_df (default full train) and transform train/test."""
    engineer = FeatureEngineer()
    engineer.fit(fit_df if fit_df is not None else train_df)
    train_feat = engineer.transform(train_df)
    test_feat = engineer.transform(test_df) if test_df is not None else None
    return train_feat, test_feat, engineer
