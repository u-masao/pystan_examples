import urllib.request
from decimal import ROUND_HALF_UP, Decimal

import arviz
import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import pandas as pd
import rdata
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
    stan_code = """
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

    stan_data = {
        "J": 8,
        "y": [28, 8, -3, 7, -1, 1, 18, 12],
        "sigma": [15, 10, 16, 11, 9, 11, 10, 18],
    }

    simulation(stan_code, stan_data)


def simulation(
    stan_code, stan_data, num_chains=4, num_samples=1000, seed=1234
):
    st.code(stan_code, language="stan")
    st.write(stan_data)
    with st.spinner("building stan code"):
        model = stan.build(stan_code, data=stan_data, random_seed=seed)
    with st.spinner("sampling now"):
        fit = model.sample(num_chains=num_chains, num_samples=num_samples)

    fig, ax = plt.subplots(2, 2)
    arviz.plot_trace(fit, axes=ax, plot_kwargs=dict(alpha=0.3))
    st.pyplot(fig)

    st.write(dir(fit))
    return fit


def load_rdata(url):
    with urllib.request.urlopen(url) as fo:
        parsed = rdata.parser.parse_data(fo.read())
    return rdata.conversion.convert(parsed)


def app_tsbook_10_1():
    dataset = load_rdata(
        "https://github.com/hagijyun/tsbook/raw/master/"
        "ArtifitialLocalLevelModel.RData"
    )

    stan_code = """
    data{
        int<lower=1> t_max;
        vector[t_max] y;

        cov_matrix[1] W;
        cov_matrix[1] V;
        real m0;
        cov_matrix[1] C0;
    }

    parameters{
        real x0;
        vector[t_max] x;
    }

    model{
        for (t in 1:t_max){
            y[t] ~ normal(x[t], sqrt(V[1, 1]));
        }
        x0 ~ normal(m0, sqrt(C0[1, 1]));
        x[1] ~ normal(x0, sqrt(W[1, 1]));
        for (t in 2:t_max){
            x[t] ~ normal(x[t-1], sqrt(W[1, 1]));
        }
    }
    """

    stan_data = {}
    stan_data["t_max"] = int(dataset["t_max"][0])
    stan_data["y"] = dataset["y"].values
    stan_data["W"] = dataset["mod"]["W"]
    stan_data["V"] = dataset["mod"]["V"]
    stan_data["m0"] = dataset["mod"]["m0"][0]
    stan_data["C0"] = dataset["mod"]["C0"]

    fit = simulation(stan_code, stan_data)
    df = fit.to_frame()
    summary_df = df.describe().T

    fig, ax = plt.subplots()
    ax.plot(dataset["y"].values)
    ax.plot(summary_df["mean"].iloc[7:].values)
    ax.fill_between(
        [x for x in range(len(summary_df) - 7)],
        summary_df["25%"].iloc[7:].values,
        summary_df["75%"].iloc[7:].values,
        alpha=0.3,
    )
    ax.grid()
    st.pyplot(fig)

    st.dataframe(df)


def main():
    apps = {
        "0_0_getting_started": app_0_0_getting_started,
        "tsbook_10_1": app_tsbook_10_1,
    }
    app_key = st.sidebar.selectbox("app", sorted(apps.keys()))
    apps[app_key]()


if __name__ == "__main__":
    main()
