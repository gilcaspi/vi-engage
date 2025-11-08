import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

from data.raw.raw_data_column_names import MEMBER_ID_COLUMN
from utils.data_loaders import get_web_visits_df, get_features_df
import matplotlib.pyplot as plt
import os
import re
from wordcloud import WordCloud


def _cluster_top_terms_and_weights(kmeans, vectorizer, top_n=150):
    terms = np.array(vectorizer.get_feature_names_out())
    centers = kmeans.cluster_centers_
    order = np.argsort(centers, axis=1)[:, ::-1]
    idx = order[:, :top_n]
    top_terms = terms[idx]
    top_weights = np.take_along_axis(centers, idx, axis=1)
    return top_terms, top_weights


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-z0-9\-_.]+", "_", s.lower())


def export_cluster_wordclouds(kmeans,
                              vectorizer,
                              cluster_names=None,
                              out_dir="cluster_wordclouds",
                              top_n=150,
                              width=1200,
                              height=900,
                              background_color="white"):
    os.makedirs(out_dir, exist_ok=True)
    top_terms, top_weights = _cluster_top_terms_and_weights(kmeans, vectorizer, top_n=top_n)
    for i in range(kmeans.n_clusters):
        freqs = {term: float(w) for term, w in zip(top_terms[i], top_weights[i]) if w > 0}
        if not freqs:
            continue
        wc = WordCloud(width=width,
                       height=height,
                       background_color=background_color,
                       prefer_horizontal=0.95,
                       collocations=False,
                       normalize_plurals=False)
        wc.generate_from_frequencies(freqs)
        title = cluster_names[i] if (cluster_names is not None and i < len(cluster_names)) else f"cluster_{i}"
        fname = os.path.join(out_dir, f"{_safe_name(title)}.png")
        plt.figure(figsize=(width/100, height/100), dpi=100)
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.savefig(fname, bbox_inches="tight", pad_inches=0)
        plt.close()
    return os.path.abspath(out_dir)


def top_terms_per_cluster(kmeans, vectorizer, top_n=15):
    terms = np.array(vectorizer.get_feature_names_out())
    centers = kmeans.cluster_centers_
    order = np.argsort(centers, axis=1)[:, ::-1]
    top_idx = order[:, :top_n]
    top_terms = terms[top_idx]
    top_weights = np.take_along_axis(centers, top_idx, axis=1)
    return top_terms, top_weights


def plot_cluster_top_terms(kmeans, vectorizer, cluster_names=None, top_n=15, cols=3, figsize=(16, 10)):
    top_terms, top_weights = top_terms_per_cluster(kmeans, vectorizer, top_n=top_n)
    n_clusters = kmeans.n_clusters
    rows = (n_clusters + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for i in range(n_clusters):
        ax = axes[i]
        terms_i = top_terms[i][::-1]
        weights_i = top_weights[i][::-1]
        ax.barh(range(len(terms_i)), weights_i)
        title = cluster_names[i] if (cluster_names is not None and i < len(cluster_names)) else f"cluster_{i}"
        ax.set_title(title)
        ax.set_yticks(range(len(terms_i)))
        ax.set_yticklabels(terms_i)
        ax.set_xlabel("TF-IDF weight")
    for j in range(n_clusters, len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    plt.show()


def _cluster_labels_to_names(kmeans, vectorizer, top_k=3):
    terms = np.array(vectorizer.get_feature_names_out())
    centroids = kmeans.cluster_centers_
    order = np.argsort(centroids, axis=1)[:, ::-1]
    names = []
    for row in order:
        top_terms = terms[row[:top_k]]
        name = "-".join(top_terms)
        name = re.sub(r"[^a-z0-9\-]+", "_", name.lower())
        names.append(name)
    return names


def _entropy_from_counts(arr):
    arr = np.asarray(arr, dtype=float)
    s = arr.sum()
    if s <= 0:
        return 0.0
    p = arr / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def add_semantic_web_category_features(web_visits_df: pd.DataFrame,
                                       member_features: pd.DataFrame,
                                       n_categories: int = 10,
                                       random_state: int = 23) -> pd.DataFrame:
    df = web_visits_df.copy()
    for c in ["url", "title", "description", "timestamp"]:
        if c not in df.columns:
            df[c] = ""
    if not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    df["__text__"] = (
        "url: " + df["url"].fillna("").astype(str) + " " +
        "title: " + df["title"].fillna("").astype(str) + " " +
        "desc: " + df["description"].fillna("").astype(str)
    ).str.lower()

    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), stop_words="english")
    X = vectorizer.fit_transform(df["__text__"])

    kmeans = MiniBatchKMeans(n_clusters=n_categories, random_state=random_state, batch_size=2048, n_init="auto")
    labels = kmeans.fit_predict(X)

    cluster_names = _cluster_labels_to_names(kmeans, vectorizer, top_k=3)
    label_to_name = {i: cluster_names[i] for i in range(n_categories)}
    df["__sem_category__"] = pd.Series(labels, index=df.index).map(label_to_name)

    df["date"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True).dt.date
    dataset_last_date = df["date"].dropna().max()

    per_member_counts = (
        df.groupby([MEMBER_ID_COLUMN, "__sem_category__"])
          .size()
          .unstack(fill_value=0)
          .reset_index()
    )
    per_member_counts.columns = [MEMBER_ID_COLUMN] + [f"web_semcat_{c}" for c in per_member_counts.columns if c != MEMBER_ID_COLUMN]

    semcat_cols = [c for c in per_member_counts.columns if c.startswith("web_semcat_")]
    counts_only = per_member_counts[semcat_cols].to_numpy()
    totals = counts_only.sum(axis=1, keepdims=True)
    shares = np.divide(counts_only, np.where(totals == 0, 1, totals))
    per_member_shares = pd.DataFrame({
        MEMBER_ID_COLUMN: per_member_counts[MEMBER_ID_COLUMN].values
    } | {c.replace("web_semcat_", "web_semcatpct_"): shares[:, i] for i, c in enumerate(semcat_cols)})

    per_member_flags = pd.DataFrame({
        MEMBER_ID_COLUMN: per_member_counts[MEMBER_ID_COLUMN].values
    } | {c.replace("web_semcat_", "web_semcat_has_"): (counts_only[:, i] > 0).astype(int) for i, c in enumerate(semcat_cols)})

    dominant_idx = counts_only.argmax(axis=1)
    dominant_name = np.array([semcat_cols[i].replace("web_semcat_", "") if totals[r, 0] > 0 else "none"
                              for r, i in enumerate(dominant_idx)])
    dominant_df = pd.DataFrame({
        MEMBER_ID_COLUMN: per_member_counts[MEMBER_ID_COLUMN].values,
        # "dominant_semcat_name": dominant_name
    })

    ent_vals = np.apply_along_axis(_entropy_from_counts, 1, counts_only)
    entropy_df = pd.DataFrame({
        MEMBER_ID_COLUMN: per_member_counts[MEMBER_ID_COLUMN].values,
        "semcat_entropy": ent_vals
    })

    per_member_time = (
        df.dropna(subset=["date"])
          .groupby([MEMBER_ID_COLUMN, "__sem_category__"])["date"]
          .agg(first_visit="min", last_visit="max", active_days="nunique")
          .reset_index()
    )
    if pd.notna(dataset_last_date):
        per_member_time["days_since_last_visit"] = (pd.to_datetime(dataset_last_date) - pd.to_datetime(per_member_time["last_visit"])).dt.days
    else:
        per_member_time["days_since_last_visit"] = np.nan

    per_member_time_counts = df.groupby([MEMBER_ID_COLUMN, "__sem_category__"]).size().rename("visits").reset_index()
    per_member_time = per_member_time.merge(per_member_time_counts, on=[MEMBER_ID_COLUMN, "__sem_category__"], how="left")
    per_member_time["freq_per_active_day"] = per_member_time["visits"] / per_member_time["active_days"].replace(0, np.nan)

    def _pivot_time(metric_name):
        pivot = (
            per_member_time
            .pivot(index=MEMBER_ID_COLUMN, columns="__sem_category__", values=metric_name)
            .fillna(0)
            .add_prefix(f"web_semcat_{metric_name}_")
            .reset_index()
        )
        pivot.columns = [MEMBER_ID_COLUMN] + [f"web_semcat_{metric_name}_{_safe_name(c.replace(f'web_semcat_{metric_name}_',''))}" for c in pivot.columns if c != MEMBER_ID_COLUMN]
        return pivot

    time_last = _pivot_time("days_since_last_visit")
    time_active = _pivot_time("active_days")
    time_freq = _pivot_time("freq_per_active_day")

    out = member_features.merge(per_member_counts, on=MEMBER_ID_COLUMN, how="left")
    out = out.merge(per_member_shares, on=MEMBER_ID_COLUMN, how="left")
    out = out.merge(per_member_flags, on=MEMBER_ID_COLUMN, how="left")
    out = out.merge(dominant_df, on=MEMBER_ID_COLUMN, how="left")
    out = out.merge(entropy_df, on=MEMBER_ID_COLUMN, how="left")
    out = out.merge(time_last, on=MEMBER_ID_COLUMN, how="left")
    out = out.merge(time_active, on=MEMBER_ID_COLUMN, how="left")
    out = out.merge(time_freq, on=MEMBER_ID_COLUMN, how="left")
    out.fillna(0, inplace=True)

    plot_cluster_top_terms(kmeans, vectorizer, cluster_names=cluster_names, top_n=15, cols=3, figsize=(18, 10))
    out_dir = export_cluster_wordclouds(kmeans, vectorizer, cluster_names=cluster_names, out_dir="cluster_wordclouds", top_n=150)
    print(f"Saved word clouds to: {out_dir}")

    return out


if __name__ == '__main__':
    member_features = add_semantic_web_category_features(
        web_visits_df=get_web_visits_df(),
        member_features=get_features_df(features_version='v1'),
        n_categories=10,
        random_state=23
    )
