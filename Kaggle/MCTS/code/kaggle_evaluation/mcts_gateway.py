''' Gateway notebook for https://www.kaggle.com/competitions/um-game-playing-strength-of-mcts-variants/
'''

import random
import time

import polars as pl

import kaggle_evaluation.core.templates



class MCTSGateway(kaggle_evaluation.core.templates.Gateway):
    ''' Expects kaggle_evaluation version '0.1.0'
    '''
    def __init__(self, data_paths: tuple[str] = None):
        super().__init__(data_paths, file_share_dir=None)
        self.set_response_timeout_seconds(10 * 60)
        self.batch_size = 100

    def unpack_data_paths(self):
        if not self.data_paths:
            self.test_path = '/kaggle/input/um-game-playing-strength-of-mcts-variants/test.csv'
            self.sample_submission_path = '/kaggle/input/um-game-playing-strength-of-mcts-variants/sample_submission.csv'
        else:
            self.test_path, self.sample_submission_path = self.data_paths

    def get_all_predictions(self):
        test = pl.read_csv(self.test_path)
        sample_submission = pl.read_csv(self.sample_submission_path)

        row_id_colname = 'Id'
        assert test.columns[0] == row_id_colname
        assert sample_submission.columns[0] == row_id_colname
        assert len(test) == test[row_id_colname].n_unique()
        assert sample_submission[row_id_colname].equals(test[row_id_colname])

        random.seed(int(time.time()))
        test = test.with_columns(pl.Series(name='new_order', values=random.sample(range(len(test)), k=len(test))))
        sample_submission = sample_submission.with_columns(test['new_order'])
        test = test.sort('new_order').drop('new_order')
        sample_submission = sample_submission.sort('new_order').drop('new_order')
        all_predictions = []
        min_index = 0
        while min_index < len(sample_submission):
            max_index = min_index + self.batch_size
            sample_sub_slice = sample_submission[min_index:max_index]
            predictions = self.predict(test[min_index:max_index], sample_sub_slice)
            predictions = pl.DataFrame(predictions)  # In case the user returned a Pandas df
            self.validate_prediction_batch(predictions, sample_sub_slice)
            all_predictions.append(predictions)
            min_index += self.batch_size
        return pl.concat(all_predictions)
