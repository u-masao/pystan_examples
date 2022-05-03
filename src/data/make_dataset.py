import logging

import click
from sklearn.datasets import fetch_openml


@click.command()
@click.argument("output_filepath", type=click.Path())
def main(**kwargs):
    logger = logging.getLogger(__name__)
    logger.info("Task start")

    dataset = fetch_openml(
        name="RAM_price", version=1, as_frame=True
    )  # data_id=40601
    dataset.frame.to_pickle(kwargs["output_filepath"])
    logger.info(f'save pickle: {kwargs["output_filepath"]}')

    logger.info("Task comlete")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
