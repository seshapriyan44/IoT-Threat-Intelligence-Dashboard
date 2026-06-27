import suppress_logs

import sys
import flwr as fl

from fl_client_base import FlowerClient

device_id = sys.argv[1]

client = FlowerClient(
    f"../dataset/{device_id}.benign.csv"
)

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=client.to_client()
)