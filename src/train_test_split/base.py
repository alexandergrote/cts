import pandas as pd
from abc import ABC, abstractmethod
from typing import Tuple

from src.util.logging import Pickler

class BaseTrainTestSplit(ABC):

    @abstractmethod
    def split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        raise NotImplementedError()
    
    def execute(self, *, data: pd.DataFrame, **kwargs) -> dict:

        print(data.shape)

        """Paper Analysis Start"""

        if 'rules' in kwargs:

            rules = kwargs['rules']

            if rules is not None:

                columns = [col for col in data.columns if '-->' in col]

                result = {}

                for column in columns:
                    result[column] = data[data[column]][self.target_name].mean() - data[self.target_name].mean()

                result = pd.Series(result)
                result.name = 'deviation_from_mean_target'

                result = rules.merge(result, left_on='index', right_index=True)

                Pickler.write(result, "rules_conf_target.pickle")

            """Paper Analysis End"""

        kwargs['x_train'], kwargs['x_test'], kwargs['y_train'], kwargs['y_test'] = self.split(data=data)

        return kwargs