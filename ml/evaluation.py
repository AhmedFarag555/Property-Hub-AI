"""
evaluate_recommender.py
=======================
Offline evaluation for the PropertyHUB Hybrid Recommendation System.

Metrics computed:
  - Precision@K, Recall@K, F1@K
  - NDCG@K  (Normalized Discounted Cumulative Gain)
  - Hit Rate@K
  - Coverage   (catalog %)
  - Intra-list Diversity (avg pairwise city/type dissimilarity)
  - Cold-start detection rate

Usage:
  cd DataForFinalProject/backend
  python evaluate_recommender.py [--k 10] [--min-interactions 2] [--test-ratio 0.2]
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
ML_DIR   = os.path.join(BASE_DIR, "ml")
for p in [BASE_DIR, ML_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── imports ────────────────────────────────────────────────────────────────────
from data_cache import cache
from smart_recommender import SmartRecommender

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

ACTION_WEIGHTS = {"view": 1, "click": 2, "favorite": 5, "contact": 10}


def build_ground_truth(interactions: pd.DataFrame, min_score: float = 3.0):
    """
    بيبني ground truth من interactions:
    - كل (user, property) بياخد weighted score
    - لو الـ score >= min_score → relevant
    """
    df = interactions.copy()
    df["score"] = df["action"].map(ACTION_WEIGHTS).fillna(1)
    agg = df.groupby(["user_id", "property_id"])["score"].sum().reset_index()
    agg["relevant"] = agg["score"] >= min_score
    return agg


def train_test_split_temporal(interactions: pd.DataFrame, test_ratio: float = 0.2):
    """
    تقسيم زمني: آخر test_ratio% من الـ interactions لكل user = test
    """
    if "created_at" not in interactions.columns:
        # لو مفيش timestamp → random split
        interactions = interactions.sample(frac=1, random_state=42)
        cutoff = int(len(interactions) * (1 - test_ratio))
        return interactions.iloc[:cutoff], interactions.iloc[cutoff:]

    interactions = interactions.sort_values("created_at")
    train_parts, test_parts = [], []

    for uid, grp in interactions.groupby("user_id"):
        n = len(grp)
        if n < 3:
            train_parts.append(grp)
            continue
        split = max(1, int(n * (1 - test_ratio)))
        train_parts.append(grp.iloc[:split])
        test_parts.append(grp.iloc[split:])

    train = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
    test  = pd.concat(test_parts,  ignore_index=True) if test_parts  else pd.DataFrame()
    return train, test


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def precision_at_k(recommended: list, relevant: set, k: int) -> float:
    if not recommended:
        return 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / k


def recall_at_k(recommended: list, relevant: set, k: int) -> float:
    if not relevant or not recommended:
        return 0.0
    hits = sum(1 for r in recommended[:k] if r in relevant)
    return hits / len(relevant)


def f1_at_k(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def ndcg_at_k(recommended: list, relevant_scores: dict, k: int) -> float:
    """
    NDCG@K مع weighted scores.
    relevant_scores: {property_id: score}
    """
    dcg  = 0.0
    idcg = 0.0

    # DCG
    for i, pid in enumerate(recommended[:k]):
        rel = relevant_scores.get(pid, 0)
        dcg += rel / np.log2(i + 2)

    # IDCG — أفضل ترتيب ممكن
    ideal = sorted(relevant_scores.values(), reverse=True)[:k]
    for i, rel in enumerate(ideal):
        idcg += rel / np.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: list, relevant: set, k: int) -> float:
    return 1.0 if any(r in relevant for r in recommended[:k]) else 0.0


def intra_list_diversity(recs: list, df_props: pd.DataFrame) -> float:
    """
    متوسط التنوع داخل القائمة بناءً على city و property_type.
    قيمة بين 0 (كل حاجة متشابهة) و 1 (كل حاجة مختلفة).
    """
    if len(recs) < 2:
        return 0.0

    prop_map = {}
    for _, row in df_props.iterrows():
        prop_map[row["property_id"]] = {
            "city":          row.get("city", ""),
            "property_type": row.get("property_type", ""),
        }

    pairs = 0
    diffs = 0
    ids   = [r["property_id"] if isinstance(r, dict) else r for r in recs]

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = prop_map.get(ids[i], {})
            b = prop_map.get(ids[j], {})
            dissim = 0
            if a.get("city")          != b.get("city"):          dissim += 0.5
            if a.get("property_type") != b.get("property_type"): dissim += 0.5
            diffs += dissim
            pairs += 1

    return diffs / pairs if pairs else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MAIN EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(k: int = 10, min_interactions: int = 2, test_ratio: float = 0.2):

    print("\n" + "═" * 60)
    print("   PropertyHUB — Recommendation System Evaluation")
    print("═" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    print("\n📦 Loading data from cache…")
    try:
        interactions = cache.get_interactions()
        df_props     = cache.properties
    except Exception as e:
        print(f"❌ Could not load cache: {e}")
        sys.exit(1)

    print(f"   Interactions : {len(interactions):,}")
    print(f"   Properties   : {len(df_props):,}")
    print(f"   Unique users : {interactions['user_id'].nunique():,}")

    # ── 2. Filter users with enough interactions ──────────────────────────────
    user_counts = interactions.groupby("user_id").size()
    valid_users = user_counts[user_counts >= min_interactions].index.tolist()
    print(f"   Users (≥{min_interactions} interactions): {len(valid_users):,}")

    if len(valid_users) == 0:
        print("❌ No users with enough interactions. Lower --min-interactions.")
        sys.exit(1)

    interactions = interactions[interactions["user_id"].isin(valid_users)]

    # ── 3. Leave-One-Out split ───────────────────────────────────────────────
    # ✅ لكل user: أعلى-weighted interaction = test item، الباقي = train
    print(f"\n✂️  Building Leave-One-Out split…")

    ACTION_W = {"contact": 10, "favorite": 5, "click": 2, "view": 1}
    interactions["weight"] = interactions["action"].map(ACTION_W).fillna(1)

    train_rows, test_rows = [], []
    user_relevant  = defaultdict(set)
    user_scores    = defaultdict(dict)

    for uid, grp in interactions.groupby("user_id"):
        grp = grp.sort_values("weight", ascending=False)
        # الـ test item = أعلى interaction (contact/favorite أولاً)
        test_item  = grp.iloc[0]
        train_part = grp.iloc[1:]

        if len(train_part) == 0:
            train_rows.append(grp)
            continue

        test_rows.append(test_item.to_frame().T)
        train_rows.append(train_part)

        pid   = int(test_item["property_id"])
        score = float(test_item["weight"])
        user_relevant[uid].add(pid)
        user_scores[uid][pid] = score

    train_df = pd.concat(train_rows, ignore_index=True) if train_rows else pd.DataFrame()
    test_df  = pd.concat(test_rows,  ignore_index=True) if test_rows  else pd.DataFrame()

    print(f"   Train: {len(train_df):,} | Test: {len(test_df):,}")

    eval_users = [u for u in valid_users if u in user_relevant]
    print(f"   Users with test item: {len(eval_users):,}")

    if not eval_users:
        print("❌ No users with test items.")
        sys.exit(1)

    # ── 5. Load recommender ───────────────────────────────────────────────────
    print("\n🤖 Loading SmartRecommender…")
    try:
        rec = SmartRecommender()
    except Exception as e:
        print(f"❌ Failed to load recommender: {e}")
        sys.exit(1)
    print("   ✅ Loaded")

    # ── Model diagnosis ───────────────────────────────────────────────────────
    model_df_size = len(rec.df)
    catalog_size_check = len(df_props)
    coverage_cap = model_df_size / catalog_size_check
    print(f"\n   Model df size     : {model_df_size:,} properties")
    print(f"   Catalog size      : {catalog_size_check:,} properties")
    print(f"   Max possible cov. : {coverage_cap:.1%}  (model can only recommend from its training set)")
    if model_df_size < catalog_size_check * 0.5:
        print(f"   ⚠️  Model trained on {coverage_cap:.1%} of catalog — retrain on full dataset for better coverage")

    # ── 6. Evaluate per user ──────────────────────────────────────────────────
    print(f"\n📊 Evaluating {len(eval_users)} users at K={k}…\n")

    results = {
        "precision": [], "recall": [], "f1": [],
        "ndcg": [], "hit_rate": [], "diversity": [],
    }
    all_recommended = set()
    strategy_counts = defaultdict(int)
    failed = 0

    # ✅ الحل: نحقن train interactions في الـ cache أثناء الـ evaluation
    # عشان full_hybrid يستبعد train items بس — ويسمح بـ test items تظهر في الـ recommendations
    original_get_interactions = cache.get_interactions

    for i, uid in enumerate(eval_users):
        if (i + 1) % max(1, len(eval_users) // 10) == 0:
            print(f"   Progress: {i+1}/{len(eval_users)} ({100*(i+1)//len(eval_users)}%)")

        try:
            # حقن train_df مكان الـ full interactions
            cache.get_interactions = lambda: train_df

            output   = rec.recommend(user_id=uid, top_n=k)

            # استعيد الأصل
            cache.get_interactions = original_get_interactions

            recs     = output.get("results", [])
            strategy = output.get("strategy", "unknown")
            strategy_counts[strategy] += 1

            rec_ids  = [
                int(r["property_id"]) if isinstance(r, dict) else int(r)
                for r in recs
            ]
            relevant = user_relevant[uid]
            scores   = user_scores[uid]


            p  = precision_at_k(rec_ids, relevant, k)
            r  = recall_at_k(rec_ids, relevant, k)
            f1 = f1_at_k(p, r)
            nd = ndcg_at_k(rec_ids, scores, k)
            hr = hit_rate_at_k(rec_ids, relevant, k)
            dv = intra_list_diversity(recs, df_props)

            results["precision"].append(p)
            results["recall"].append(r)
            results["f1"].append(f1)
            results["ndcg"].append(nd)
            results["hit_rate"].append(hr)
            results["diversity"].append(dv)

            all_recommended.update(rec_ids)

        except Exception as e:
            cache.get_interactions = original_get_interactions
            failed += 1

    # ── 7. Catalog Coverage ───────────────────────────────────────────────────
    catalog_size = len(df_props)
    coverage     = len(all_recommended) / catalog_size if catalog_size else 0

    # ── 8. Print results ──────────────────────────────────────────────────────
    def mean(lst): return np.mean(lst) if lst else 0.0

    print("\n" + "═" * 60)
    print(f"   RESULTS  (K = {k})")
    print("═" * 60)
    print(f"   Users evaluated   : {len(eval_users) - failed:,}  (failed: {failed})")
    print()
    print(f"   Precision@{k:<3}     : {mean(results['precision']):.4f}")
    print(f"   Recall@{k:<3}        : {mean(results['recall']):.4f}")
    print(f"   F1@{k:<3}            : {mean(results['f1']):.4f}")
    print(f"   NDCG@{k:<3}          : {mean(results['ndcg']):.4f}")
    print(f"   Hit Rate@{k:<3}      : {mean(results['hit_rate']):.4f}")
    print()
    print(f"   Intra-list Div.   : {mean(results['diversity']):.4f}  (0=homogeneous, 1=diverse)")
    print(f"   Catalog Coverage  : {coverage:.2%}  ({len(all_recommended):,} / {catalog_size:,} items seen)")
    print()
    print("   Strategy distribution:")
    for s, cnt in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / max(len(eval_users), 1)
        print(f"     {s:<20} : {cnt:>4}  ({pct:.1f}%)")
    print("═" * 60)

    # ── 9. Analysis & Diagnosis ──────────────────────────────────────────────
    print("\n" + "─" * 60)
    print("   DIAGNOSIS")
    print("─" * 60)

    # Coverage diagnosis
    model_df_size = len(rec.df)
    max_coverage  = model_df_size / catalog_size if catalog_size else 1
    if coverage < 0.05:
        if model_df_size < catalog_size * 0.5:
            print(f"   ⚠️  Low coverage ({coverage:.1%}) — model trained on only {max_coverage:.1%} of catalog")
            print(f"       → ROOT CAUSE: Retrain content model on full {catalog_size:,} properties")
            print(f"       → Current model only knows {model_df_size:,} properties")
        else:
            print(f"   ⚠️  Low coverage ({coverage:.1%}) — recommender favors same items repeatedly")
            print(f"       → Consider increasing diversity in CandidateGenerator (top_k)")

    # Diversity diagnosis
    avg_div = mean(results["diversity"])
    if avg_div < 0.1:
        print(f"   ⚠️  Low diversity ({avg_div:.3f}) — most recs same city/type")
        print(f"       → Increase lambda_div in SmartRecommender.diversify()")

    # Hit rate diagnosis
    avg_hr = mean(results["hit_rate"])
    if avg_hr < 0.05:
        print(f"   ⚠️  Low hit rate ({avg_hr:.3f}) — model struggles to find relevant items")
        print(f"       → Check if content model features cover test-set properties")
    elif avg_hr < 0.2:
        print(f"   🟡 Moderate hit rate ({avg_hr:.3f}) — room for improvement")
    else:
        print(f"   ✅ Good hit rate ({avg_hr:.3f})")

    # NDCG diagnosis
    avg_ndcg = mean(results["ndcg"])
    if avg_ndcg > 0.1:
        print(f"   ✅ NDCG@{k} = {avg_ndcg:.4f} — ranking quality acceptable")
    else:
        print(f"   ⚠️  NDCG@{k} = {avg_ndcg:.4f} — ranking needs improvement")

    # Data sparsity
    avg_interactions = interactions.groupby("user_id").size().mean()
    print(f"\n   Data sparsity:")
    print(f"     Avg interactions/user : {avg_interactions:.1f}")
    print(f"     Total interactions    : {len(interactions):,}")
    print(f"     Total users           : {interactions['user_id'].nunique():,}")
    if avg_interactions < 5:
        print(f"   ⚠️  Very sparse data — cold-start dominates, metrics will be low")
        print(f"       → Collect more user interactions before expecting high metrics")

    print("─" * 60)

    if len(strategy_counts) > 1:
        print("\n   Note: Metrics averaged across all strategies.")
        print("   Use --min-interactions 5 to isolate full_hybrid users only.")

    print()
    return {
        "precision": mean(results["precision"]),
        "recall":    mean(results["recall"]),
        "f1":        mean(results["f1"]),
        "ndcg":      mean(results["ndcg"]),
        "hit_rate":  mean(results["hit_rate"]),
        "diversity": mean(results["diversity"]),
        "coverage":  coverage,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PropertyHUB recommender")
    parser.add_argument("--k",                type=int,   default=10,  help="Cutoff K (default: 10)")
    parser.add_argument("--min-interactions", type=int,   default=2,   help="Min interactions per user (default: 2)")
    parser.add_argument("--test-ratio",       type=float, default=0.2, help="Test split ratio (default: 0.2)")
    args = parser.parse_args()

    evaluate(
        k=args.k,
        min_interactions=args.min_interactions,
        test_ratio=args.test_ratio,
    )