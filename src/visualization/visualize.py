from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
import stan
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


def download_and_visualize():
    df = pd.read_pickle("data/raw/dataset.pickle")
    df = preprocess(df)
    st.dataframe(df)
    st.dataframe(df.describe())
    st.line_chart(df)


def app_0_0_getting_started():
    schools_code = """
    data {
      int<lower=0> J;         // number of schools
      real y[J];              // estimated treatment effects
      real<lower=0> sigma[J]; // standard error of effect estimates
    }
    parameters {
      real mu;                // population treatment effect
      real<lower=0> tau;      // standard deviation in treatment effects
      vector[J] eta;          // unscaled deviation from mu by school
    }
    transformed parameters {
      vector[J] theta = mu + tau * eta;        // school treatment effects
    }
    model {
      target += normal_lpdf(eta | 0, 1);       // prior log-density
      target += normal_lpdf(y | theta, sigma); // log-likelihood
    }
    """

    schools_data = {
        "J": 8,
        "y": [28, 8, -3, 7, -1, 1, 18, 12],
        "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
    }

    with st.spinner("building stan code"):
        posterior = stan.build(schools_code, data=schools_data)
    with st.spinner("sampling now"):
        fit = posterior.sample(num_chains=4, num_samples=1000)
    eta = fit["eta"]  # array with shape (8, 4000)
    df = fit.to_frame()  # pandas `DataFrame, requires pandas

    st.dataframe(df)
    st.table(eta.T)

    st.line_chart(eta.T)


def main():
    apps = {"0_0_getting_started": app_0_0_getting_started}
    app_key = st.sidebar.selectbox("app", apps.keys())
    apps[app_key]()


if __name__ == "__main__":
    main()
