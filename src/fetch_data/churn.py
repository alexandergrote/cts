import pandas as pd

from pydantic import BaseModel, field_validator
from typing import Optional
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler

from src.fetch_data.base import BaseDataLoader
from src.util.caching import pickle_cache
from src.util.constants import Directory


class ChurnDataloader(BaseModel, BaseDataLoader):

    """
    Data taken from https://www.coveo.com/en/ailabs/shopper-intent-prediction-from-clickstream-e-commerce-data-with-minimal-browsing-information
    See also: https://github.com/coveooss/shopper-intent-prediction-nature-2020
    """

    path: str
    resampling_fraction: Optional[float] = None
    max_length: int = 100
    resampling_fraction: Optional[float] = None

    class Config:
        arbitrary_types_allowed=True

    @field_validator('path')
    def _set_path(cls, v):

        filepath = str(Directory.DATA / v)

        return filepath

    @pickle_cache(ignore_caching=True)
    def execute(self) -> dict:

        data = pd.read_csv(self.path)

        data.dropna(subset=['product_action'], inplace=True)

        sequences = data.groupby('session_id_hash')['product_action'].apply(list)
        
        class_labels_mapping = sequences.apply(lambda x: 'purchase' in x).to_dict()

        trimmed_sequences = sequences.apply(lambda x: x[:x.index('purchase')] if 'purchase' in x else x)

        trimmed_sequences = trimmed_sequences[trimmed_sequences.apply(len).between(2,80)]

        records = []

        for session_id, sequences in tqdm(trimmed_sequences.items(), total=len(trimmed_sequences)):

            time_idx = 0

            for sequence in sequences:

                record = {
                    'session_id': session_id,
                    'product_action': sequence,
                    'time': time_idx,
                    'label': class_labels_mapping[session_id]
                }

                records.append(record)

                time_idx += 1

        data = pd.DataFrame.from_records(records)

        # resample if argument is provided
        if self.resampling_fraction is not None:

            n_old = data['session_id'].nunique()

            sampler = RandomUnderSampler(
                sampling_strategy=self.resampling_fraction,
                random_state=42
            )

            tmp = data[['session_id', 'label']].drop_duplicates()
            tmp['label'] = tmp['label'].astype(int)
            ids, _ = sampler.fit_resample(tmp.drop(columns=['label']), tmp['label'])

            data = data[data['session_id'].isin(ids['session_id'].to_list())]
            
            n_new = data['session_id'].nunique()

            print(f"Resampled from {n_old} to {n_new} samples")
        
        
        return {'data': data}


if __name__ == '__main__':

    ts_data = ChurnDataloader(
        path="release_10_23_2020.csv"
    ).execute()

    print(ts_data['data'])
