
import os
import sys

import kaggle_evaluation.core.templates

sys.path.append(os.path.dirname(os.path.realpath(__file__)))
import mcts_gateway


class MCTSInferenceServer(kaggle_evaluation.core.templates.InferenceServer):
    def _get_gateway_for_test(self, data_paths=None):
        return mcts_gateway.MCTSGateway(data_paths)
