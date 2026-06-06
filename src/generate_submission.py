import pandas as pd

SEPARATOR = "||"


def generate_submission(
    original_path,
    filled_path,
    output_path
):

    original = pd.read_csv(
        original_path
    )

    filled = pd.read_csv(
        filled_path
    )

    feature_cols = [
        c
        for c in original.columns
        if c != "datetime"
    ]

    rows = []

    for col in feature_cols:

        missing_mask = (
            original[col].isna()
        )

        for idx in original.index[
            missing_mask
        ]:

            dt = original.loc[
                idx,
                "datetime"
            ]

            uid = (
                f"{dt}"
                f"{SEPARATOR}"
                f"{col}"
            )

            value = filled.loc[
                idx,
                col
            ]

            rows.append({
                "id": uid,
                "value": value
            })

    submission = pd.DataFrame(rows)

    submission = (
        submission
        .sort_values("id")
        .reset_index(drop=True)
    )

    submission.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved -> {output_path}"
    )