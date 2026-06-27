import flwr as fl

from fl_data_loader import load_client_data
from model import create_autoencoder


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, csv_path):
        self.X = load_client_data(csv_path)
        self.model = create_autoencoder()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)

        self.model.fit(
            self.X,
            self.X,
            epochs=1,
            batch_size=256,
            verbose=0
        )

        return (
            self.model.get_weights(),
            len(self.X),
            {}
        )

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)

        loss = self.model.evaluate(
            self.X,
            self.X,
            verbose=0
        )

        return loss, len(self.X), {"loss": loss}