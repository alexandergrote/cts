import optuna
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


class LSTMBenchmark(BaseModel, BaseProcessModel, TorchMixin):

    model: Any
    evaluator: Any

    batch_size: int
    num_epochs: int
    learning_rate: float
    patience: int

    optimizer: Optional[torch.optim.Adam] = None
    loss_fn: nn.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    classes: np.ndarray = np.empty(shape=(0,))

    event_mapping: Optional[dict] = None

    class Config:
        arbitrary_types_allowed=True

    @field_validator('model', 'evaluator')
    def _set_model(cls, v):
        return DynamicImport.import_class_from_dict(dictionary=v)
    
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
        
        # record classes
        self.classes = torch.Tensor(np.unique(y_train)).to(self.device)

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

    
    def predict(self, x_test: pd.DataFrame, **kwargs):

        sigmoid_values = self.predict_proba(x_test)
        y_pred = np.argmax(sigmoid_values, axis=1)
        y_pred = y_pred.reshape(-1)
        
        return y_pred
    
    def predict_proba(self, x_test: pd.DataFrame, **kwargs):
        
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
            


if __name__ == '__main__':

    import yaml
    from src.fetch_data.synthetic import DataLoader
    from src.util.constants import Directory, replace_placeholder_in_dict

    # get constants
    with open(Directory.CONFIG / 'constants\synthetic.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    # get synthetic data
    with open(Directory.CONFIG / 'fetch_data\synthetic.yaml', 'r') as file:
        config = yaml.safe_load(file)['params']


    for _, value in cfg.items():

        placeholder = value['placeholder']
        replacement = value['value']

        config = replace_placeholder_in_dict(
            dictionary=config,
            placeholder=placeholder,
            replacement=replacement
        )

    data_loader = DataLoader(**config)
    data = data_loader.execute()['data']

    # get model config
    with open(Directory.CONFIG / 'model\lstm.yaml', 'r') as file:
        config = yaml.safe_load(file)['params']

    model = LSTMBenchmark(**config)

    mapping = {event: i+1 for i, event in enumerate(data['event_column'].unique())}
    data['event_column'] = data['event_column'].map(mapping)

    # get sequences from data
    data.sort_values(by='timestamp', inplace=True)
    sequences = data.groupby('id_column')['event_column'].apply(list).to_list()
    targets = data.groupby('id_column')['target'].apply(lambda x: np.unique(x)[0]).to_list()
    
    # split train test
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        sequences, targets, test_size=0.2, random_state=42
    )
    
    model.fit(
        x_train=pd.DataFrame(x_train),
        y_train=pd.Series(y_train)
    )

    y_pred_proba = model.predict_proba(pd.DataFrame(x_test))
    y_pred = model.predict(pd.DataFrame(x_test))

    # get evaluation config
    with open(Directory.CONFIG / 'evaluation\ml.yaml', 'r') as file:
        cfg = yaml.safe_load(file)

    evaluator = DynamicImport.import_class_from_dict(cfg)

    result = evaluator.evaluate(
        y_pred=y_pred,
        y_pred_proba=y_pred_proba, 
        y_test=y_test
    )
    
    print(result['metrics'])


    

