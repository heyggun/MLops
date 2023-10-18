# TODO: Write proper metric function

from sklearn.metrics import (
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
)

"""
class EvalPrediction:
    '''
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    '''
    
    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs
"""


def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}
