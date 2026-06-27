import suppress_logs
import flwr as fl

from model import create_autoencoder

print("Starting Flower Server...")

global_weights = None


class SaveModelStrategy(fl.server.strategy.FedAvg):

    def aggregate_fit(
        self,
        server_round,
        results,
        failures,
    ):

        global global_weights

        aggregated = super().aggregate_fit(
            server_round,
            results,
            failures
        )

        if aggregated is not None:

            parameters, _ = aggregated

            global_weights = fl.common.parameters_to_ndarrays(
                parameters
            )

            print(
                f"\nSaved weights from Round {server_round}"
            )

        return aggregated


strategy = SaveModelStrategy(
    min_fit_clients=9,
    min_evaluate_clients=9,
    min_available_clients=9
)

fl.server.start_server(
    server_address="127.0.0.1:8080",
    config=fl.server.ServerConfig(
        num_rounds=3
    ),
    strategy=strategy,
)

print("\nTraining Finished")

if global_weights is not None:

    model = create_autoencoder()

    model.set_weights(global_weights)

    model.save(
        "../models/federated_autoencoder.keras"
    )

    print(
        "\nFederated model saved:"
    )
    print(
        "../models/federated_autoencoder.keras"
    )