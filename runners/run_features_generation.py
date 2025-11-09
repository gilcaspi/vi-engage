import os
import argparse

from data.features import FEATURES_DIRECTORY_PATH
from preprocessing.features_generation import generate_features
from preprocessing.web_visits_classification import add_semantic_web_category_features
from utils.data_loaders import get_web_visits_df


def get_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate features for the dataset.")
    parser.add_argument(
        "--features_version",
        type=str,
        default="v2",
        help="Version of the features to generate (e.g., v1, v2, etc.)"
    )
    parser.add_argument(
        "--n_categories",
        type=int,
        default=10,
        help="Number of semantic web categories (K-Means clusters)."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=23,
        help="Random seed for reproducibility."
    )
    return parser


if __name__ == "__main__":
    args_parser = get_arg_parser()
    args = args_parser.parse_args()

    print(f"Generating features (version={args.features_version})")
    features_df = generate_features()

    print("Adding semantic web category features")
    features_df = add_semantic_web_category_features(
        web_visits_df=get_web_visits_df(),
        member_features=features_df,
        n_categories=args.n_categories,
        random_state=args.random_state
    )

    output_file_path = os.path.join(
        FEATURES_DIRECTORY_PATH,
        f"generated_features_{args.features_version}.csv"
    )

    os.makedirs(FEATURES_DIRECTORY_PATH, exist_ok=True)
    features_df.to_csv(output_file_path, index=False)

    print(f"Features {args.features_version} saved to: {output_file_path}")
