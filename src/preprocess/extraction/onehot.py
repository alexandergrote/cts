import pandas as pd
from pydantic import BaseModel

from src.preprocess.base import BaseFeatureEncoder


class OneHotEncoder(BaseModel, BaseFeatureEncoder):

    id_column: str
    feature_column: str
    target_column: str  

    def _encode(self, *, data: pd.DataFrame, **kwargs) -> dict:

        data_copy = data.copy()

        # one hot encode data
        df_pivot = data_copy[[self.id_column, self.feature_column]].drop_duplicates().pivot_table(index=self.id_column, columns=self.feature_column, aggfunc='size')
        df_pivot = ~df_pivot.isna()
        df_pivot = df_pivot.astype(int)
        df_pivot.index.name = self.id_column
        df_pivot.reset_index(inplace=True)

        # merge maleware back to data
        df_maleware = data_copy[[self.id_column, self.target_column]].drop_duplicates()
        df_pivot = df_pivot.merge(df_maleware, left_on=self.id_column, right_on=self.id_column, how='inner')

        assert df_pivot.shape[0] == data_copy[self.id_column].nunique()

        kwargs['data'] = df_pivot.drop(columns=[self.id_column], errors='ignore')
        
        return kwargs
