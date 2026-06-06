import numpy as np
import pandas as pd

from scipy.interpolate import (
    PchipInterpolator,
    CubicSpline
)


def extract_strike(col_name: str) -> int:
    digits = ''.join(
        ch for ch in col_name
        if ch.isdigit()
    )

    return int(digits[-5:])


def get_option_columns(df):

    return [
        c
        for c in df.columns
        if c not in [
            "datetime",
            "underlying_price"
        ]
    ]


def fill_missing_iv(df):

    df = df.copy()

    iv_cols = get_option_columns(df)

    strikes = np.array([
        extract_strike(col)
        for col in iv_cols
    ])

    sort_idx = np.argsort(strikes)

    strikes = strikes[sort_idx]
    iv_cols = [iv_cols[i] for i in sort_idx]

    # ==================================
    # ROW-WISE SMILE RECONSTRUCTION
    # ==================================

    for row_idx in range(len(df)):

        spot = float(
            df.loc[row_idx,
                   "underlying_price"]
        )

        row = (
            df.loc[row_idx, iv_cols]
            .values
            .astype(float)
        )

        missing = np.isnan(row)

        if not missing.any():
            continue

        observed = ~missing

        if observed.sum() < 4:
            continue

        x_obs = np.log(
            strikes[observed] / spot
        )

        y_obs = row[observed]

        x_target = np.log(
            strikes[missing] / spot
        )

        try:

            pchip = PchipInterpolator(
                x_obs,
                y_obs,
                extrapolate=True
            )

            cubic = CubicSpline(
                x_obs,
                y_obs,
                extrapolate=True
            )

            p_pred = pchip(x_target)

            c_pred = cubic(x_target)

            pred = (
                0.5 * p_pred +
                0.5 * c_pred
            )

            row[missing] = pred

            df.loc[
                row_idx,
                iv_cols
            ] = row

        except Exception:
            pass

    # safety fallback

    df[iv_cols] = (
        df[iv_cols]
        .ffill(axis=1)
        .bfill(axis=1)
    )

    return df