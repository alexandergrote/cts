import numpy as np
from pydantic import BaseModel, field_validator
from typing import List, Union, Callable, Optional

from src.evaluation.base import BaseEvaluator
from src.util.dynamic_import import DynamicImport


class Evaluator(BaseModel, BaseEvaluator):
    metrics: Union[List[str], List[Callable]]
    metrics_proba: Optional[Union[List[str], List[Callable]]] = None

    @field_validator("metrics", "metrics_proba")
    def _get_sklearn_metric(cls, v):

        if v is None:
            return []

        return [DynamicImport._get_class_obj(metric) for metric in v]

    @staticmethod
    def _check_if_metric_is_in_list(metric: str, metric_list: List[Callable]):

        # get all metric names from list
        names = [el.__name__ for el in metric_list]

        return metric in names

    @staticmethod
    def _get_metric_from_name(name: str, metric_list: List[Callable]) -> Callable:

        # get all metric names from list
        names = [el.__name__ for el in metric_list]

        idx = names.index(name)

        return metric_list[idx]

    def get_metric_names(self) -> List[str]:

        metric_list = self.metrics.copy()
        metric_list += self.metrics_proba.copy()

        return [metric.__name__ for metric in metric_list]

    def evaluate(self, y_pred: np.ndarray, y_test: np.ndarray, **kwargs) -> dict:

        # get all metrics
        metrics = self.get_metric_names()

        # result placeholder
        result = {}

        # use names in for loop to ensure consistency
        for metric in metrics:

            if self._check_if_metric_is_in_list(metric, self.metrics):
                metric_callable = self._get_metric_from_name(metric, self.metrics)
                result[metric] = metric_callable(y_pred=y_pred, y_true=y_test)

            elif self._check_if_metric_is_in_list(metric, self.metrics_proba):
                y_pred_proba = kwargs['y_pred_proba'][:, 1]
                metric_callable = self._get_metric_from_name(metric, self.metrics_proba)
                result[metric] = metric_callable(y_score=y_pred_proba, y_true=y_test)

        kwargs['metrics'] = result
        kwargs['y_pred'] = y_pred
        kwargs['y_test'] = y_test

        return kwargs
