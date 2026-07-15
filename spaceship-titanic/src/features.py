"""
features.py — shared feature engineering for Spaceship Titanic.

Imported by the modeling scripts so train and test get identical treatment.
Design choices are driven by the EDA (see TUTORIAL.md Step 1):

- Decode compound IDs: PassengerId -> Group/GroupSize; Cabin -> Deck/Num/Side.
- Deterministic imputation using the CryoSleep<->spend link.
- Group-level features (people travel — and are transported — together).
"""
import numpy as np
import pandas as pd

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
CATEGORICAL = ["HomePlanet", "Destination", "Deck", "Side", "CryoSleep", "VIP"]


def _fill_by_group_mode(df, col, by):
    """Fill missing values of `col` with the most common value among rows that
    share the same `by` key (e.g. the same travel Group). Leak-safe: uses only
    feature columns, never the target."""
    def mode_or_nan(s):
        m = s.mode()
        return m.iloc[0] if len(m) else np.nan
    grp_mode = df.groupby(by)[col].transform(mode_or_nan)
    return df[col].fillna(grp_mode)


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    """Return (X_train, y_train, X_test) with engineered, model-ready features.

    train/test are the raw CSVs. We concatenate them for ID-structure features
    (group size, family size) — this uses only the ID columns, never the target,
    so it does not leak.
    """
    y = train["Transported"].astype(int)

    both = pd.concat([train.drop(columns=["Transported"]), test], ignore_index=True)
    n_train = len(train)

    # --- Decode PassengerId = gggg_pp -------------------------------------
    both["Group"] = both["PassengerId"].str.split("_").str[0]
    both["GroupSize"] = both["Group"].map(both["Group"].value_counts())
    both["IsAlone"] = (both["GroupSize"] == 1).astype(int)

    # --- Decode Cabin = deck/num/side -------------------------------------
    cab = both["Cabin"].str.split("/", expand=True)
    both["Deck"] = cab[0]
    both["Num"] = pd.to_numeric(cab[1], errors="coerce")
    both["Side"] = cab[2]

    # --- Surname (family) from Name ---------------------------------------
    both["Surname"] = both["Name"].str.split().str[-1]
    both["FamilySize"] = both["Surname"].map(both["Surname"].value_counts())

    # --- Group-based imputation of categoricals (v3) ----------------------
    # HomePlanet is constant within a travel group and within a family, so a
    # missing value can be recovered exactly from groupmates/relatives. Cabin
    # deck & side are usually shared within a group too. This replaces global
    # guesses with local, structurally-justified fills.
    both["HomePlanet"] = _fill_by_group_mode(both, "HomePlanet", "Group")
    both["HomePlanet"] = _fill_by_group_mode(both, "HomePlanet", "Surname")
    both["Destination"] = _fill_by_group_mode(both, "Destination", "Group")
    both["Deck"] = _fill_by_group_mode(both, "Deck", "Group")
    both["Side"] = _fill_by_group_mode(both, "Side", "Group")

    # --- Deterministic CryoSleep <-> spend logic --------------------------
    both["TotalSpend"] = both[SPEND].sum(axis=1, min_count=1)
    spent = both["TotalSpend"] > 0
    # Anyone who spent money was awake.
    both.loc[spent, "CryoSleep"] = both.loc[spent, "CryoSleep"].fillna(False)
    # Asleep passengers spend nothing -> fill their missing amenities with 0.
    asleep = both["CryoSleep"] == True  # noqa: E712
    both.loc[asleep, SPEND] = both.loc[asleep, SPEND].fillna(0)
    # Recompute total after filling.
    both["TotalSpend"] = both[SPEND].sum(axis=1, min_count=1)
    both["HasSpend"] = (both["TotalSpend"] > 0).astype("float")

    # --- Log-transform skewed spend ---------------------------------------
    for c in SPEND:
        both[c + "_log"] = np.log1p(both[c])
    both["TotalSpend_log"] = np.log1p(both["TotalSpend"])

    # --- Simple imputation for the rest -----------------------------------
    for c in SPEND + [c + "_log" for c in SPEND] + ["TotalSpend", "TotalSpend_log"]:
        both[c] = both[c].fillna(0)
    both["Age"] = both["Age"].fillna(both["Age"].median())
    both["Num"] = both["Num"].fillna(both["Num"].median())

    # --- Spend structure (v2) --------------------------------------------
    # EDA/community insight: "luxury" amenities (RoomService, Spa, VRDeck)
    # behave differently from "basics" (FoodCourt, ShoppingMall) w.r.t. the
    # target, so split them out. Also count how many categories were used.
    both["LuxurySpend"] = both[["RoomService", "Spa", "VRDeck"]].sum(axis=1)
    both["BasicSpend"] = both[["FoodCourt", "ShoppingMall"]].sum(axis=1)
    both["LuxurySpend_log"] = np.log1p(both["LuxurySpend"])
    both["BasicSpend_log"] = np.log1p(both["BasicSpend"])
    both["NumSpentCats"] = (both[SPEND] > 0).sum(axis=1).astype("float")
    for c in SPEND:
        both[c + "_spent"] = (both[c] > 0).astype("float")

    # --- Age structure (v2) ----------------------------------------------
    both["IsChild"] = (both["Age"] < 13).astype("float")

    # --- Cabin region (v2): bucket cabin number along the ship ------------
    both["CabinRegion"] = (both["Num"] // 300).astype("float")

    # --- Group-level aggregates (v2, leak-safe: no target used) -----------
    # People travel in groups and tend to share a fate; summarizing a
    # passenger's group (before we drop it) injects that context per row.
    grp = both.groupby("Group")
    both["GroupSpendMean"] = grp["TotalSpend"].transform("mean")
    both["GroupCryoRate"] = grp["CryoSleep"].transform(
        lambda s: s.astype("float").mean())
    both["GroupAgeMean"] = grp["Age"].transform("mean")

    # Categoricals: encode as pandas 'category' codes (LightGBM-friendly,
    # NaN -> its own code -1 which trees can split on).
    for c in CATEGORICAL:
        both[c] = both[c].astype("category").cat.codes

    feature_cols = (
        ["Age", "GroupSize", "IsAlone", "FamilySize", "Num", "HasSpend",
         "TotalSpend", "TotalSpend_log"]
        + SPEND + [c + "_log" for c in SPEND]
        # v2 additions
        + ["LuxurySpend", "BasicSpend", "LuxurySpend_log", "BasicSpend_log",
           "NumSpentCats", "IsChild", "CabinRegion",
           "GroupSpendMean", "GroupCryoRate", "GroupAgeMean"]
        + [c + "_spent" for c in SPEND]
        + CATEGORICAL
    )

    X = both[feature_cols]
    X_train = X.iloc[:n_train].reset_index(drop=True)
    X_test = X.iloc[n_train:].reset_index(drop=True)
    return X_train, y, X_test, feature_cols
