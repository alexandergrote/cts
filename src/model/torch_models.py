import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pydantic import BaseModel, field_validator
from typing import Any, Optional

from src.model.base import BaseProcessModel
from src.model.util.torch_early_stopping import EarlyStopping
from src.model.util.torch_util import TorchMixin
from src.util.dynamic_import import DynamicImport
from src.util.constants import Directory

DEVICE = torch.device('cpu')

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

if torch.backends.mps.is_available():
    DEVICE = torch.device('mps')


class LSTMBenchmark(BaseModel, BaseProcessModel, TorchMixin):

    model: Any

    batch_size: int
    num_epochs: int
    learning_rate: float
    patience: int

    optimizer: Optional[torch.optim.Adam] = None
    loss_fn: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    device: torch.device = DEVICE

    event_mapping: Optional[dict] = None

    class Config:
        arbitrary_types_allowed=True
        extra = 'forbid'

    @field_validator('model')
    def _set_model(cls, v):
        return DynamicImport.import_class_from_dict(dictionary=v)
    
    def get_annotated_hyperparameters(self) -> dict:

        list_of_attributes = [
            'batch_size', 'num_epochs', 'learning_rate'
        ]

        hyperparameters = {
            attribute: self.__annotations__[attribute] for attribute in list_of_attributes
        }

        # add hidden dimensions as hyperparameter
        hyperparameters['model'] = {'params': {'hidden_size': int}}
            
        return hyperparameters
    
    def _calculate_loss(
        self, model, data, labels
    ):

        model = model.to(self.device)
        data = data.to(self.device)
        labels = labels.to(self.device)

        clf_output = model(data)
        loss_clf = self.loss_fn(clf_output, labels)

        return loss_clf

    def train(
        self,
        model,
        device,
        train_loader,
        validation_loader,
        optimizer,
        epoch,
    ):

        model.train()

        train_loss = []

        for batch_idx, (data, labels) in enumerate(train_loader):

            data, labels = data.to(device), labels.to(device)
            labels = labels.type(torch.FloatTensor)
            optimizer.zero_grad()
            loss = self._calculate_loss(
                model=model,
                data=data,
                labels=labels,
            )
            loss.backward()
            optimizer.step()

            if batch_idx % 20 == 0:
                print(
                    "Epoch {} Iteration {}: Loss = {}".format(
                        epoch, batch_idx, loss
                    )
                )

            train_loss.append(loss.cpu().detach().numpy())

        # calculate validation loss
        model.eval()

        valid_loss = []

        with torch.no_grad():

            for batch_idx, (data, labels) in enumerate(validation_loader):
                labels = labels.type(torch.FloatTensor)
                loss = self._calculate_loss(
                    model=model,
                    data=data,
                    labels=labels
                )
                valid_loss.append(loss.cpu().detach().numpy())

            return np.mean(np.array(train_loss)), np.mean(np.array(valid_loss))

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series, **kwargs):

        # init model
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

        checkpoint_dir = Directory.OUTPUT_DIR / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / str(self.model.__class__.__name__ + '.pt')

        # prepare early stopping
        early_stopping = EarlyStopping(
            patience=self.patience,
            delta=0,
            path=str(checkpoint_file),
            verbose=True,
        )

        # create data loader
        train_loader = self.get_data_loader(
            x=x_train.values, y=y_train.values, batch_size=self.batch_size
        )
        valid_loader = self.get_data_loader(
            x=x_train.values, y=y_train.values, batch_size=self.batch_size
        )

         # epoch results
        epoch_train_results = []
        epoch_valid_results = []

        for epoch in range(1, self.num_epochs + 1):

            train_results, valid_results = self.train(
                model=self.model,
                device=self.device,
                train_loader=train_loader,
                validation_loader=valid_loader,
                optimizer=self.optimizer,
                epoch=epoch,
            )

            epoch_train_results.append(train_results)
            epoch_valid_results.append(valid_results)

            early_stopping(valid_results, self.model)

            if early_stopping.early_stop:
                break

        # load the last checkpoint with the best model
        self.model.load_state_dict(
            torch.load(str(checkpoint_file))
        )

        self.plot_loss(
            epoch_train_results, 
            epoch_valid_results,
            filename=str(self.model.__class__.__name__ + '__loss.png')
        )

    def _predict(self, x_test: pd.DataFrame, **kwargs):

        sigmoid_values = self._predict_proba(x_test)
        y_pred = np.argmax(sigmoid_values, axis=1)
        y_pred = y_pred.reshape(-1)
        
        return y_pred
    
    def _predict_proba(self, x_test: pd.DataFrame, **kwargs):
        
        x = torch.Tensor(x_test.values).to("cpu")
        x = self.prepare_x(x)

        self.model = self.model.to("cpu")

        with torch.no_grad():

            self.model.eval()

            outputs = self.model(x)
            sigmoid_values = torch.sigmoid(outputs)

            # standardize output to match sklearn's predict_proba
            sigmoid_values = sigmoid_values.cpu().detach().numpy()
            sigmoid_values = np.hstack([1 - sigmoid_values, sigmoid_values])

            return sigmoid_values  

