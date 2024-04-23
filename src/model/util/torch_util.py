import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from typing import Union

import torch
from sklearn.model_selection import StratifiedKFold

class StratifiedBatchSampler:
    """
    Stratified batch sampling
    Provides equal representation of target classes in each batch
    Taken from https://discuss.pytorch.org/t/how-to-enable-the-dataloader-to-sample-from-each-class-with-equal-probability/911/7
    """
    def __init__(self, y, batch_size, shuffle=True):
        if torch.is_tensor(y):
            y = y.numpy()
        assert len(y.shape) == 1, 'label array must be 1D'
        n_batches = max(2, int(len(y) / batch_size))

        self.skf = StratifiedKFold(n_splits=n_batches, shuffle=shuffle, random_state=42)
        self.X = torch.randn(len(y),1).numpy()
        self.y = y
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.skf.random_state = torch.randint(0,int(1e8),size=()).item()
        for train_idx, test_idx in self.skf.split(self.X, self.y):
            yield test_idx

    def __len__(self):
        return len(self.y)


class TorchMixin:

    def prepare_x(self, x: np.ndarray) -> torch.Tensor:
            
        x = torch.Tensor(x)

        padded_sequences = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=0)

        filled_tensor = torch.where(torch.isnan(padded_sequences), torch.tensor(0), padded_sequences)

        filled_tensor = filled_tensor.int()


        return filled_tensor
    
    def prepare_y(self, y: np.ndarray) -> torch.Tensor:

        y = torch.Tensor(y)
        y = y.float()
        y = y.view(-1, 1)

        return y

    def get_dataset(self, x: np.ndarray, y: np.ndarray) -> TensorDataset:

        x = self.prepare_x(x)
        y = self.prepare_y(y)

        dataset = TensorDataset(x, y)

        return dataset

    def get_data_loader(
        self, x: np.ndarray, y: np.ndarray, batch_size: int
    ) -> DataLoader:

        dataset = self.get_dataset(x=x, y=y)

        dataset_loader = DataLoader(
            dataset,
            batch_sampler=StratifiedBatchSampler(
                y, batch_size=batch_size
            )
        )

        return dataset_loader
    
    def _map_labels(self, y: Union[np.ndarray, torch.Tensor], mapping: dict) -> np.ndarray:

        y_relabelled = None

        if isinstance(y, np.ndarray):
            y_relabelled = np.copy(y)
        
        if isinstance(y, torch.Tensor):
            y_relabelled = torch.clone(y)


        for label, index in mapping.items():
            y_relabelled[y == label] = index

        return y_relabelled

    def plot_loss(self, train_results, valid_results, filename: str):
        epochs = np.array(range(len(train_results))) + 1
        plt.figure()
        plt.plot(epochs, train_results, label="train loss")
        plt.plot(epochs, valid_results, label="validation loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(loc="upper right")
        plt.xticks(epochs)
        plt.savefig(filename)