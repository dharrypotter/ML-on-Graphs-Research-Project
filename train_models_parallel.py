import logging
import pickle
from logging.handlers import TimedRotatingFileHandler
from multiprocessing import Pool, cpu_count
from pathlib import Path

import click
import numpy as np

from eelp.utils.parallel_utils import chunk, process_graphs


@click.command()
@click.option(
    "--input-path",
    "-i",
    "input_data_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option("--output-path", "-o", required=True, type=click.Path(dir_okay=True))
@click.option("--n-samples", "num_samples", default=10000, type=click.INT, show_default=True)
@click.option(
    "--sampling-method", type=click.Choice(["rs", "rswi", "hnes"]), default="rs", show_default=True
)
@click.option("--num-procs", type=click.INT, default=-1)
def main(input_data_path, output_path, num_samples, sampling_method, num_procs):
    # Determine the number of concurrent processes to launch
    procs = num_procs if num_procs > 0 else cpu_count()
    output_path = Path(output_path).resolve()
    proc_ids = list(range(0, procs))
    # Load input data
    logging.info("Grabbing Input Data")
    with open(input_data_path, "rb") as f:
        df = pickle.load(f)
    # Create graph meta dictionaries
    input_graphs = []
    logging.info("Creating output directories")
    for row in df.itertuples(index=False):
        graph_out_path = output_path / f"{int(row.network_index)}"
        graph_out_path.mkdir(exist_ok=True, parents=True)
        input_graphs.append(
            {
                "level_0": int(row.level_0),
                "network_index": int(row.network_index),
                "network_name": row.network_name,
                "num_nodes": int(row.number_nodes),
                "num_edges": int(row.number_edges),
                "edge_list": row.edges_id,
                "output_path": graph_out_path,
            }
        )
    # Divide the graphs into chunks to be consumed by each process
    num_graphs_per_proc = len(input_graphs) / float(procs)
    num_graphs_per_proc = int(np.ceil(num_graphs_per_proc))
    chunked_graphs = chunk(input_graphs, num_graphs_per_proc)
    # initialize the list of payloads
    payloads = []

    # loop over the set chunked graph sets
    for i, graph_set in enumerate(chunked_graphs):
        # construct the path to the output intermediary file for the
        # current process
        pickle_output_path = Path(output_path) / f"proc_{i}.pickle"

        # construct a dictionary of data for the payload, then add it
        # to the payloads list
        data = {
            "id": i,
            "input_graphs": graph_set,
            "output_path": pickle_output_path,
            "num_samples": num_samples,
            "sampling_method": sampling_method,
        }
        payloads.append(data)

    # Now we use multiprocessing
    logger.info("Launching pool using {} processes...".format(procs))
    pool = Pool(processes=procs)
    pool.map(process_graphs, payloads)
    pool.close()
    pool.join()
    logger.info("Multiprocessing complete")


if __name__ == "__main__":
    # Finding project home
    project_dir = Path(__file__).resolve().parents[2]
    # load_dotenv(find_dotenv())
    # Setting up logging configuration for the module

    # Create a custom logger
    logger = logging.getLogger()
    # Setting root logger level to the lowest so all info messages can be logged to the file
    logger.setLevel(logging.DEBUG)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = TimedRotatingFileHandler(
        project_dir.joinpath("logs", "graph_process.log"),
        when="midnight",
    )
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    main()
