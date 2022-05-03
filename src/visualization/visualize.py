from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
import streamlit as st


def preprocess(df):
    df["year"] = df["date"].astype(int)
    df["month"] = (1 + df["date"].mod(1) * 12).map(
        lambda x: int(
            Decimal(str(x)).quantize(Decimal("0"), rounding=ROUND_HALF_UP)
        )
    )
    df["day"] = 1
    df = df[df["month"] < 13]
    df["ts"] = pd.to_datetime(df[["year", "month", "day"]])
    df = df.set_index("ts").loc["2010":, "price"]
    monthly_df = pd.DataFrame(
        index=pd.date_range(
            start=df.index.min(), end=df.index.max(), freq="MS"
        )
    )
    merged_df = monthly_df.merge(
        df, how="left", left_index=True, right_index=True
    )
    return merged_df.interpolate(method="linear")


def main():
    df = pd.read_pickle("data/raw/dataset.pickle")
    df = preprocess(df)
    st.dataframe(df)
    st.dataframe(df.describe())
    st.line_chart(df)


if __name__ == "__main__":
    main()
