

class TestDataset(unittest.TestCase):

    def setUp(self) -> None:

        self.id_column = 'id_column'
        self.time_column = 'time_column'
        self.event_column = 'event_column'
        self.class_column = 'class_column'

        self.raw_data = pd.DataFrame({
            self.id_column: ['a', 'a', 'a', 'b', 'b'],
            self.time_column: [i for i in range(5)],
            self.event_column: [1,2,3,5,4],
            self.class_column: [1, 0, 1, 0, 1]
        })

    def test_get_sequences(self):

        dataset = Dataset(
            id_column=self.id_column,
            event_column=self.event_column,
            time_column=self.time_column,
            class_column=self.class_column,
            raw_data=self.raw_data
        )

        sequences = dataset.get_sequences()

        self.assertIsInstance(sequences, list)
        self.assertTrue(all([isinstance(el, AnnotatedSequence) for el in sequences]))
        self.assertEqual([el.sequence_values for el in sequences], [['1','2', '3'], ['5','4']])
        self.assertEqual([el.id_value for el in sequences], ['a', 'b'])
        
    def test_get_sequences_without_classes(self):

        data_copy = self.raw_data.copy(deep=True)
        data_copy.drop(columns=self.class_column, inplace=True)

        dataset = Dataset(
            id_column=self.id_column,
            event_column=self.event_column,
            time_column=self.time_column,
            raw_data=self.raw_data
        )

        sequences = dataset.get_sequences()

        self.assertIsInstance(sequences, list)
        self.assertTrue(all([isinstance(el, AnnotatedSequence) for el in sequences]))
        self.assertEqual([el.sequence_values for el in sequences], [['1', '2', '3'], ['5', '4']])
        self.assertEqual([el.id_value for el in sequences], ['a', 'b'])
