import pandas as pd
import streamlit as st


def main():
    df = pd.read_pickle("data/interim/dataset.pickle")
    st.pyplot(df)


if __name__ == "__main__":
    main()
