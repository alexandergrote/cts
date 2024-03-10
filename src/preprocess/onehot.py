import pandas as pd
from pydantic import BaseModel

from src.preprocess.base import BasePreprocessor


class OneHotEncoder(BaseModel, BasePreprocessor):

    id_column: str
    feature_column: str  

    def execute(self, *, event: pd.DataFrame, **kwargs) -> dict:

        data_copy = event.copy()

        # one hot encode data
        df_pivot = data_copy[['id', 'value']].drop_duplicates().pivot_table(index='id', columns='value', aggfunc='size')
        df_pivot = ~df_pivot.isna()
        df_pivot = df_pivot.astype(int)
        df_pivot.index.name = 'id'
        df_pivot.reset_index(inplace=True)

        # merge maleware back to data
        df_maleware = data_copy[['id', 'malware']].drop_duplicates()
        df_pivot = df_pivot.merge(df_maleware, left_on='id', right_on='id', how='inner')

        assert df_pivot.shape[0] == data_copy['id'].nunique()

        kwargs['data'] = df_pivot
        
        return kwargs
