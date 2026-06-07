from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import random
from typing import List
import numpy as np
import pandas as pd
from artrec.utils.common import ensure_dir
from artrec.data.io import write_csv

USER_SEGMENTS = [
    "loyal_collector",
    "casual_browser",
    "trend_seeker",
    "budget_buyer",
    "explorer",
]
SURFACES = ["home_feed", "similar_item", "email_push", "search_browse"]
DEVICES = ["ios", "android", "web"]
COUNTRIES = ["GB", "US", "DE", "FR", "NL"]
START_DATE = datetime(2026, 1, 1)

# Conservative funnel calibration for a synthetic art marketplace demo.
# These keep the simulator useful for ML training while avoiding implausible
# "perfect" ecommerce outcomes such as many purchases in a single session.
CLICK_RATE_SCALE = 0.35
SAVE_RATE_SCALE = 0.65
CART_RATE_SCALE = 0.22
IMMEDIATE_PURCHASE_SCALE = 0.10
DELAYED_PURCHASE_SCALE = 0.08


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9))


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


@dataclass
class SimulationConfig:
    n_users: int = 500
    n_days: int = 45
    max_sessions_per_day: int = 3
    slate_size: int = 20
    candidate_pool_size: int = 200
    random_seed: int = 42


class BehaviorSimulator:
    def __init__(self, catalog_df: pd.DataFrame, config: SimulationConfig):
        self.catalog_df = catalog_df.copy()
        self.config = config
        self.rnd = random.Random(config.random_seed)
        np.random.seed(config.random_seed)
        self.item_index = self.catalog_df.set_index("item_id", drop=False)
        self.item_ids = self.catalog_df["item_id"].tolist()
        self.users_df = self._generate_users()
        self.user_history = {
            uid: {
                "seen_items": [],
                "clicked_items": [],
                "saved_items": [],
                "purchased_items": [],
                "not_interested_items": [],
                "not_interested_artists": [],
                "not_interested_styles": [],
                "artist_counts": {},
                "style_counts": {},
            }
            for uid in self.users_df["user_id"].tolist()
        }
        self.item_impressions = {item_id: 0 for item_id in self.item_ids}
        self.item_clicks = {item_id: 0 for item_id in self.item_ids}
        self.item_purchases = {item_id: 0 for item_id in self.item_ids}
        self.pending_purchase_intents: List[dict] = []

    def _generate_users(self) -> pd.DataFrame:
        segment_params = {
            "loyal_collector": dict(
                budget_mean=900,
                price_sensitivity=0.45,
                novelty=0.35,
                repeat=0.75,
                activity=0.75,
                conv=0.55,
                save=0.48,
            ),
            "casual_browser": dict(
                budget_mean=250,
                price_sensitivity=1.00,
                novelty=0.40,
                repeat=0.45,
                activity=0.45,
                conv=0.14,
                save=0.20,
            ),
            "trend_seeker": dict(
                budget_mean=450,
                price_sensitivity=0.80,
                novelty=0.82,
                repeat=0.40,
                activity=0.85,
                conv=0.25,
                save=0.35,
            ),
            "budget_buyer": dict(
                budget_mean=150,
                price_sensitivity=1.45,
                novelty=0.22,
                repeat=0.52,
                activity=0.55,
                conv=0.20,
                save=0.24,
            ),
            "explorer": dict(
                budget_mean=340,
                price_sensitivity=0.90,
                novelty=0.95,
                repeat=0.25,
                activity=0.95,
                conv=0.22,
                save=0.42,
            ),
        }
        latent_dim = len(self.catalog_df["embedding"].iloc[0])
        styles = sorted(self.catalog_df["style"].unique().tolist())
        rows = []
        for i in range(self.config.n_users):
            segment = self.rnd.choice(USER_SEGMENTS)
            p = segment_params[segment]
            taste = normalize(np.random.normal(0, 1, latent_dim))
            preferred_styles = self.rnd.sample(styles, k=self.rnd.randint(1, 3))
            rows.append(
                {
                    "user_id": f"user_{i:04d}",
                    "segment": segment,
                    "country": self.rnd.choice(COUNTRIES),
                    "device_pref": self.rnd.choice(DEVICES),
                    "budget_mean": max(
                        60.0,
                        np.random.normal(p["budget_mean"], p["budget_mean"] * 0.25),
                    ),
                    "price_sensitivity": max(
                        0.1, np.random.normal(p["price_sensitivity"], 0.15)
                    ),
                    "novelty_preference": min(
                        1.0, max(0.0, np.random.normal(p["novelty"], 0.10))
                    ),
                    "repeat_tolerance": min(
                        1.0, max(0.0, np.random.normal(p["repeat"], 0.10))
                    ),
                    "activity_level": min(
                        1.6, max(0.1, np.random.normal(p["activity"], 0.15))
                    ),
                    "conversion_propensity": min(
                        1.0, max(0.01, np.random.normal(p["conv"], 0.08))
                    ),
                    "save_propensity": min(
                        1.0, max(0.01, np.random.normal(p["save"], 0.08))
                    ),
                    "preferred_styles": "|".join(preferred_styles),
                    "taste_vector": taste.tolist(),
                }
            )
        return pd.DataFrame(rows)

    def _popularity_score(self, item_id: str) -> float:
        impressions = max(1, self.item_impressions[item_id])
        ctr = self.item_clicks[item_id] / impressions
        p_rate = self.item_purchases[item_id] / impressions
        base = float(self.item_index.loc[item_id, "popularity_prior"])
        return 0.55 * base + 0.25 * ctr + 0.20 * p_rate

    def _user_item_affinity(self, user_row: pd.Series, item_row: pd.Series) -> float:
        history = self.user_history[user_row["user_id"]]
        taste = np.array(user_row["taste_vector"], dtype=float)
        item_vec = np.array(item_row["embedding"], dtype=float)
        latent = cosine_sim(taste, item_vec)
        style_bonus = (
            0.25
            if item_row["style"] in str(user_row["preferred_styles"]).split("|")
            else 0.0
        )
        seen_count = history["seen_items"].count(item_row["item_id"])
        repeat_penalty = min(
            1.5, seen_count * (1.0 - float(user_row["repeat_tolerance"])) * 0.25
        )
        artist_seen = history["artist_counts"].get(item_row["artist_id"], 0)
        artist_penalty = min(1.0, max(0, artist_seen - 2) * 0.08)
        price_penalty = float(user_row["price_sensitivity"]) * max(
            0,
            (float(item_row["price"]) - float(user_row["budget_mean"]))
            / max(50, float(user_row["budget_mean"])),
        )
        novelty_bonus = (
            float(user_row["novelty_preference"]) * float(item_row["freshness"]) * 0.4
        )
        return (
            1.8 * latent
            + style_bonus
            + 0.5 * float(item_row["quality"])
            + novelty_bonus
            - repeat_penalty
            - artist_penalty
            - price_penalty
        )

    def _session_count(self, user_row: pd.Series, day_idx: int) -> int:
        weekday = (START_DATE + timedelta(days=day_idx)).weekday()
        weekend_bonus = 0.25 if weekday >= 5 else 0.0
        lam = max(0.05, float(user_row["activity_level"]) + weekend_bonus)
        return min(self.config.max_sessions_per_day, np.random.poisson(lam))

    def _sample_candidates(self, user_row: pd.Series) -> list[str]:
        scored = []
        for item_id in self.item_ids:
            item_row = self.item_index.loc[item_id]
            if int(item_row["availability"]) == 0:
                continue
            affinity = self._user_item_affinity(user_row, item_row)
            popularity = self._popularity_score(item_id)
            score = (
                0.60 * affinity
                + 0.25 * popularity
                + 0.15 * float(item_row["freshness"])
                + np.random.normal(0, 0.15)
            )
            scored.append((item_id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_pref = [x[0] for x in scored[: int(self.config.candidate_pool_size * 0.7)]]
        random_pool = self.rnd.sample(
            self.item_ids,
            k=min(len(self.item_ids), int(self.config.candidate_pool_size * 0.3)),
        )
        merged = list(dict.fromkeys(top_pref + random_pool))
        return merged[: self.config.candidate_pool_size]

    def _rank_slate(
        self, user_row: pd.Series, candidate_ids: list[str], surface: str
    ) -> list[str]:
        ranked = []
        for item_id in candidate_ids:
            item_row = self.item_index.loc[item_id]
            affinity = self._user_item_affinity(user_row, item_row)
            popularity = self._popularity_score(item_id)
            surface_bonus = (
                0.15 * float(item_row["freshness"]) if surface == "email_push" else 0.0
            )
            score = (
                0.65 * affinity
                + 0.25 * popularity
                + 0.10 * float(item_row["freshness"])
                + surface_bonus
            )
            ranked.append((item_id, score))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in ranked[: self.config.slate_size]]

    def _choose_surface(self, user_row: pd.Series) -> str:
        weights = {
            "home_feed": 0.55,
            "similar_item": 0.15,
            "email_push": 0.10,
            "search_browse": 0.20,
        }
        if user_row["segment"] == "loyal_collector":
            weights["email_push"] += 0.05
        if user_row["segment"] == "explorer":
            weights["search_browse"] += 0.08
        surfaces = list(weights.keys())
        probs = np.array([weights[s] for s in surfaces], dtype=float)
        probs /= probs.sum()
        return np.random.choice(surfaces, p=probs)

    def _position_exam_prob(self, pos: int) -> float:
        return max(0.10, 0.95 * math.exp(-0.18 * pos))

    def _drift_user_preferences(self, user_id: str):
        idx = self.users_df.index[self.users_df["user_id"] == user_id][0]
        history = self.user_history[user_id]
        recent_clicked = history["clicked_items"][-10:]
        if not recent_clicked:
            return
        vectors = np.array(
            [self.item_index.loc[item_id, "embedding"] for item_id in recent_clicked],
            dtype=float,
        )
        avg_vec = normalize(vectors.mean(axis=0))
        taste = np.array(self.users_df.at[idx, "taste_vector"], dtype=float)
        self.users_df.at[idx, "taste_vector"] = normalize(
            0.93 * taste + 0.07 * avg_vec
        ).tolist()

    def _process_delayed(self, current_day: datetime) -> list[dict]:
        emitted, remaining = [], []
        for intent in self.pending_purchase_intents:
            if intent["trigger_date"].date() <= current_day.date():
                if self.rnd.random() < intent["purchase_prob"]:
                    item_row = self.item_index.loc[intent["item_id"]]
                    self.item_purchases[intent["item_id"]] += 1
                    self.user_history[intent["user_id"]]["purchased_items"].append(
                        intent["item_id"]
                    )
                    emitted.append(
                        {
                            "event_type": "purchase",
                            "timestamp": current_day,
                            "user_id": intent["user_id"],
                            "item_id": intent["item_id"],
                            "session_id": intent["session_id"],
                            "revenue": float(item_row["price"]),
                            "margin": float(item_row["margin"]),
                            "delayed": 1,
                        }
                    )
            else:
                remaining.append(intent)
        self.pending_purchase_intents = remaining
        return emitted

    def run(self):
        impressions, interactions, conversions = [], [], []
        session_counter = 0
        for day_idx in range(self.config.n_days):
            current_day = START_DATE + timedelta(days=day_idx)
            conversions.extend(self._process_delayed(current_day))
            for _, user_row in self.users_df.iterrows():
                n_sessions = self._session_count(user_row, day_idx)
                for _ in range(n_sessions):
                    session_counter += 1
                    session_id = f"session_{session_counter:06d}"
                    session_has_purchase = False
                    surface = self._choose_surface(user_row)
                    device = (
                        user_row["device_pref"]
                        if self.rnd.random() < 0.7
                        else self.rnd.choice(DEVICES)
                    )
                    ts = current_day.replace(
                        hour=self.rnd.choice([8, 12, 18, 20, 22]),
                        minute=self.rnd.randint(0, 59),
                    )
                    candidates = self._sample_candidates(user_row)
                    slate = self._rank_slate(user_row, candidates, surface)
                    for pos, item_id in enumerate(slate):
                        item_row = self.item_index.loc[item_id]
                        history = self.user_history[user_row["user_id"]]
                        self.item_impressions[item_id] += 1
                        history["seen_items"].append(item_id)
                        popularity = self._popularity_score(item_id)
                        affinity = self._user_item_affinity(user_row, item_row)
                        examined = self.rnd.random() < sigmoid(
                            1.5 * self._position_exam_prob(pos) + 0.3 * popularity - 0.5
                        )
                        clicked = 0
                        dwell_seconds = 0
                        saved = 0
                        add_to_cart = 0
                        purchased = 0
                        not_interested = 0
                        not_interested_reason = "none"
                        if examined:
                            click_prob = CLICK_RATE_SCALE * sigmoid(
                                1.0 * affinity
                                + 0.25 * popularity
                                + np.random.normal(0, 0.5)
                                - 1.1
                            )
                            clicked = int(self.rnd.random() < click_prob)
                            if not clicked:
                                seen_count = history["seen_items"].count(item_id)
                                artist_seen = history["artist_counts"].get(
                                    item_row["artist_id"], 0
                                )
                                style_seen = history["style_counts"].get(
                                    item_row["style"], 0
                                )
                                price_gap = max(
                                    0.0,
                                    (
                                        float(item_row["price"])
                                        - float(user_row["budget_mean"])
                                    )
                                    / max(80, float(user_row["budget_mean"])),
                                )
                                poor_match = max(0.0, -affinity)
                                negative_prob = min(
                                    0.22,
                                    0.025
                                    + 0.035 * poor_match
                                    + 0.035 * min(price_gap, 2.0)
                                    + 0.018 * max(0, artist_seen - 3)
                                    + 0.014 * max(0, style_seen - 4)
                                    + 0.020 * max(0, seen_count - 2),
                                )
                                not_interested = int(self.rnd.random() < negative_prob)
                                if not_interested:
                                    reason_scores = {
                                        "poor_match": poor_match,
                                        "too_expensive": price_gap,
                                        "repeated_artist": max(0, artist_seen - 3),
                                        "repeated_style": max(0, style_seen - 4),
                                        "already_seen": max(0, seen_count - 2),
                                    }
                                    not_interested_reason = max(
                                        reason_scores, key=reason_scores.get
                                    )
                                    if reason_scores[not_interested_reason] <= 0:
                                        not_interested_reason = "poor_match"
                                    history["not_interested_items"].append(item_id)
                                    history["not_interested_artists"].append(
                                        item_row["artist_id"]
                                    )
                                    history["not_interested_styles"].append(
                                        item_row["style"]
                                    )
                        if clicked:
                            self.item_clicks[item_id] += 1
                            history["clicked_items"].append(item_id)
                            dwell_mean = max(
                                4,
                                8
                                + 18 * max(0, affinity)
                                + 3 * float(item_row["quality"]),
                            )
                            dwell_seconds = int(
                                np.random.gamma(
                                    shape=2.5, scale=max(1.0, dwell_mean / 4.0)
                                )
                            )
                            save_prob = SAVE_RATE_SCALE * sigmoid(
                                0.9 * affinity
                                + 0.4 * float(user_row["save_propensity"])
                                + 0.15 * float(item_row["freshness"])
                                - 0.1
                                * (
                                    float(item_row["price"])
                                    / max(100, float(user_row["budget_mean"]))
                                )
                                + np.random.normal(0, 0.3)
                                - 1.0
                            )
                            saved = int(self.rnd.random() < save_prob)
                            if saved:
                                history["saved_items"].append(item_id)
                            cart_prob = CART_RATE_SCALE * sigmoid(
                                1.0 * affinity
                                + 0.25 * float(item_row["quality"])
                                + 0.25 * saved
                                - 0.9
                                * float(user_row["price_sensitivity"])
                                * max(
                                    0,
                                    (
                                        float(item_row["price"])
                                        - float(user_row["budget_mean"])
                                    )
                                    / max(80, float(user_row["budget_mean"])),
                                )
                                + np.random.normal(0, 0.35)
                                - 1.2
                            )
                            add_to_cart = int(self.rnd.random() < cart_prob)
                            purchase_prob = sigmoid(
                                1.0 * affinity
                                + 0.45 * float(user_row["conversion_propensity"])
                                + 0.55 * add_to_cart
                                + 0.15 * saved
                                - 1.1
                                * float(user_row["price_sensitivity"])
                                * max(
                                    0,
                                    (
                                        float(item_row["price"])
                                        - float(user_row["budget_mean"])
                                    )
                                    / max(80, float(user_row["budget_mean"])),
                                )
                                + np.random.normal(0, 0.4)
                                - 1.6
                            )
                            if (
                                not session_has_purchase
                                and self.rnd.random()
                                < purchase_prob * IMMEDIATE_PURCHASE_SCALE
                            ):
                                purchased = 1
                                session_has_purchase = True
                                self.item_purchases[item_id] += 1
                                history["purchased_items"].append(item_id)
                                conversions.append(
                                    {
                                        "event_type": "purchase",
                                        "timestamp": ts,
                                        "user_id": user_row["user_id"],
                                        "item_id": item_id,
                                        "session_id": session_id,
                                        "revenue": float(item_row["price"]),
                                        "margin": float(item_row["margin"]),
                                        "delayed": 0,
                                    }
                                )
                            elif add_to_cart == 1 or saved == 1:
                                self.pending_purchase_intents.append(
                                    {
                                        "trigger_date": current_day
                                        + timedelta(
                                            days=self.rnd.choice([1, 2, 3, 5, 7])
                                        ),
                                        "user_id": user_row["user_id"],
                                        "item_id": item_id,
                                        "session_id": session_id,
                                        "purchase_prob": purchase_prob
                                        * DELAYED_PURCHASE_SCALE,
                                    }
                                )
                            history["artist_counts"][item_row["artist_id"]] = (
                                history["artist_counts"].get(item_row["artist_id"], 0)
                                + 1
                            )
                            history["style_counts"][item_row["style"]] = (
                                history["style_counts"].get(item_row["style"], 0) + 1
                            )

                        row = {
                            "session_id": session_id,
                            "timestamp": ts,
                            "user_id": user_row["user_id"],
                            "surface": surface,
                            "device": device,
                            "country": user_row["country"],
                            "position": pos,
                            "item_id": item_id,
                            "artist_id": item_row["artist_id"],
                            "style": item_row["style"],
                            "price": float(item_row["price"]),
                            "margin": float(item_row["margin"]),
                            "freshness": float(item_row["freshness"]),
                            "availability": int(item_row["availability"]),
                            "logging_policy_score": round(
                                self._position_exam_prob(pos), 4
                            ),
                            "popularity_score_pre": round(popularity, 4),
                            "affinity_score": round(affinity, 4),
                            "examined": int(examined),
                            "clicked": int(clicked),
                            "dwell_seconds": dwell_seconds,
                            "saved": saved,
                            "add_to_cart": add_to_cart,
                            "purchased_immediate": purchased,
                            "not_interested": not_interested,
                            "not_interested_reason": not_interested_reason,
                        }
                        impressions.append(row)
                        if (
                            clicked
                            or saved
                            or add_to_cart
                            or purchased
                            or not_interested
                        ):
                            interactions.append(row.copy())
                self._drift_user_preferences(user_row["user_id"])
        return (
            self.users_df.copy(),
            self.catalog_df.copy(),
            pd.DataFrame(impressions),
            pd.DataFrame(interactions),
            pd.DataFrame(conversions),
        )

    def save_outputs(
        self,
        users_df,
        catalog_df,
        impressions_df,
        interactions_df,
        conversions_df,
        output_dir,
    ):
        ensure_dir(output_dir)
        paths = {
            "users": write_csv(users_df, output_dir / "users.csv"),
            "catalog": write_csv(catalog_df, output_dir / "catalog.csv"),
            "impressions": write_csv(impressions_df, output_dir / "impressions.csv"),
            "interactions": write_csv(interactions_df, output_dir / "interactions.csv"),
            "conversions": write_csv(conversions_df, output_dir / "conversions.csv"),
        }
        return paths
