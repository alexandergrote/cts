import pandas as pd

from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional

from src.preprocess.extraction.ts_features import PrefixSpan, RuleClassifier
from src.preprocess.base import BaseFeatureEncoder
from src.util.caching import PickleCacheHandler, hash_dataframe
from src.util.custom_logging import console

class SPM(BaseModel, BaseFeatureEncoder):

    id_columns: List[str]
    event_column: str
    time_column: str
    target_column: str
    splitting_symbol: str = ' --> '
    itertool_threshold: Optional[int] = None

    def _get_unique_rules(self, *, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
            
        # work on copy
        rules = data.copy(deep=True)

        # init column names
        rule_column = 'index'
        sorting_column = 'confidence'
        confidence_column = 'confidence'

        # sort by abs delta confidence to make drop duplicates easier
        # keep the rules with the higher absolute delta confidence
        rules[sorting_column] = rules[confidence_column].abs()
        rules.sort_values(by=sorting_column, ascending=False, inplace=True)

        # prepare rules for vectorized comparison
        rules_as_list = rules[rule_column].str.split(self.splitting_symbol)
        rules_as_str = rules_as_list.apply(lambda x: f'{self.splitting_symbol}'.join(list(filter(None, x))))

        # drop duplicates
        mask = rules_as_str.duplicated(keep='first')
        rules = rules[~mask]

        # drop sorting column
        rules.drop(columns=[sorting_column], inplace=True)

        return rules
    
    def _apply_rule_to_ts(self, *, rules: pd.DataFrame, event: pd.DataFrame, **kwargs):

        hash_str = '__'.join([hash_dataframe(rules), hash_dataframe(event)])

        cache_handler = PickleCacheHandler(
            filepath=Path('rule_clf') / f'{hash_str}.pickle'
        )

        res = cache_handler.read()

        if res is not None:
            return res


        # work on copy
        rules_copy = rules.copy(deep=True)
        event_copy = event.copy(deep=True)

        # check data validity
        assert 'index' in rules_copy.columns
        assert self.time_column in event_copy.columns
        assert self.event_column in event_copy.columns
        for col in self.id_columns:
            assert col in event_copy.columns

        # to apply the rules, we need to make sure that the incoming dataframe is ordered
        event_copy = event_copy.sort_values(by=self.time_column)
        event_copy.reset_index(drop=True, inplace=True)

        # create sequence for each id
        column_name = 'event_sequences'
        event_grouped = event_copy.groupby(self.id_columns)
        event_sequences = event_grouped.apply(lambda x:  list(x[self.event_column]))
        event_sequences.name = column_name
        event_sequences_df = event_sequences.to_frame()

        dataframes = [event_sequences_df.reset_index()]

        for _, row in tqdm(rules_copy.iterrows(), total=len(rules_copy)):

            rule = list(filter(None, row['index'].split(self.splitting_symbol)))
            rule_clf = RuleClassifier(rule=rule, _cache={})
            
            sequences = event_sequences_df[column_name].to_list()
            result = rule_clf.apply_rules(sequences)

            dataframe = pd.DataFrame({row['index']: result})

            dataframes.append(dataframe)

        event_sequences_df = pd.concat(dataframes, axis=1)

        # fill missing values
        event_sequences_df.fillna(False, inplace=True)

        # drop column name
        event_sequences_df.drop(column_name, axis=1, inplace=True)
        event_sequences_df.reset_index(inplace=True)

        # write to cache
        cache_handler.write(event_sequences_df)

        return event_sequences_df

    def _encode(self, *, data: pd.DataFrame, **kwargs) -> dict:

        data_copy = data.copy(deep=True)

        prefix_span = PrefixSpan(
            id_columns=self.id_columns,
            event_column=self.event_column,
            splitting_symbol=self.splitting_symbol,
            itertool_threshold=self.itertool_threshold
        )

        console.log('Extracting rules from data...')

        rules = prefix_span.execute(data=data_copy)

        console.log('Creating unique rules...')
        rules['index'] = rules.apply(lambda x: f"{x['antecedent']}{self.splitting_symbol*2}{x['precedent']}", axis=1)
        
        rules = self._get_unique_rules(data=rules)

        console.log('Applying rules to data...')
        data_one_hot = self._apply_rule_to_ts(rules=rules, event=data_copy)

        data_class = data_copy[self.id_columns +  [self.target_column]].drop_duplicates()
        data_one_hot = data_one_hot.merge(data_class, right_on=self.id_columns, left_on=self.id_columns)
        
        data_one_hot.drop(columns=self.id_columns + ['index'], inplace=True)

        return dict(data=data_one_hot)

if __name__ == '__main__':

    import hashlib

    case_name = "test_" + hashlib.sha1(str.encode(str(datetime.now()))).hexdigest()
    sequence = ['a', 'b', 'c', 'd', 'e']

    df = pd.DataFrame(sequence, columns=['events'])
    df['case_id'] = 1
    df['timestamp'] = pd.date_range('2020-01-01', periods=len(sequence), freq='D')

    spm = SPM(id_columns=['case_id'], event_column='events', time_column='timestamp')
    result = spm.execute(data=df, case_name=case_name)

    data = result['data']
    print(data)

