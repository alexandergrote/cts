import pandas as pd

from pydantic import BaseModel, field_validator, ConfigDict
from typing import Optional
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler

from src.fetch_data.base import BaseDataLoader
from src.util.caching import pickle_cache
from src.util.constants import Directory
from src.util.datasets import Dataset, DatasetSchema


class ChurnDataloader(BaseModel, BaseDataLoader):

    """
    Data taken from https://www.coveo.com/en/ailabs/shopper-intent-prediction-from-clickstream-e-commerce-data-with-minimal-browsing-information
    See also: https://github.com/coveooss/shopper-intent-prediction-nature-2020
    """

    path: str
    resampling_fraction: Optional[float] = None
    max_length: int = 100
    resampling_fraction: Optional[float] = None
    n_samples: Optional[int] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    @field_validator('path')
    def _set_path(cls, v):

        filepath = str(Directory.DATA / v)

        return filepath

    @pickle_cache(ignore_caching=True, cachedir=Directory.CACHING_DIR / 'churn')
    def get_data(self) -> Dataset:

        data = pd.read_csv(self.path)

        data.dropna(subset=['product_action'], inplace=True)

        sequences = data.groupby('session_id_hash')['product_action'].apply(list)
        
        class_labels_mapping = sequences.apply(lambda x: 'purchase' in x).to_dict()

        trimmed_sequences = sequences.apply(lambda x: x[:x.index('purchase')] if 'purchase' in x else x)

        trimmed_sequences = trimmed_sequences[trimmed_sequences.apply(len).between(5,155)]

        records = []

        counter = 0

        for session_id, sequences in tqdm(trimmed_sequences.items(), total=len(trimmed_sequences)):

            time_idx = 0

            for sequence in sequences:

                record = {
                    DatasetSchema.id_column: session_id,
                    DatasetSchema.event_column: sequence,
                    DatasetSchema.time_column: time_idx,
                    DatasetSchema.class_column: class_labels_mapping[session_id]
                }

                records.append(record)

                time_idx += 1

            counter += 1

            if self.n_samples is not None and counter >= self.n_samples:
                break

        data = pd.DataFrame.from_records(records)

        # resample if argument is provided
        if self.resampling_fraction is not None:

            n_old = data[DatasetSchema.id_column].nunique()

            sampler = RandomUnderSampler(
                sampling_strategy=self.resampling_fraction,
                random_state=42
            )

            tmp = data[[DatasetSchema.id_column, DatasetSchema.class_column]].drop_duplicates()
            tmp[DatasetSchema.class_column] = tmp[DatasetSchema.class_column].astype(int)
            ids, _ = sampler.fit_resample(tmp.drop(columns=[DatasetSchema.class_column]), tmp[DatasetSchema.class_column])

            data = data[data[DatasetSchema.id_column].isin(ids[DatasetSchema.id_column].to_list())]
            
            n_new = data[DatasetSchema.id_column].nunique()

            print(f"Resampled from {n_old} to {n_new} samples")

        # format result
        data[DatasetSchema.class_column] = data[DatasetSchema.class_column].astype('int64')

        
        return data


if __name__ == '__main__':

    ts_data = ChurnDataloader(
        path="release_10_23_2020.csv",
        n_samples=1000
    ).execute()

    print(ts_data['data'])
