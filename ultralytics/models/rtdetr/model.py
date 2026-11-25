from ultralytics.engine.model import Model
from ultralytics.nn.tasks import DEFAULT_MODEL, RTDETRDetectionModel

from .predict import RTDETRPredictor
from .train import RTDETRTrainer
from .val import RTDETRValidator


class RTDETR(Model):

    def __init__(self, model=DEFAULT_MODEL) -> None:
        if model and str(model).split('.')[-1] not in ('pt', 'yaml', 'yml'):
            raise NotImplementedError('RT-DETR only supports creating from *.pt, *.yaml, or *.yml files.')
        super().__init__(model=model, task='detect')

    @property
    def task_map(self) -> dict:
        return {
            'detect': {
                'predictor': RTDETRPredictor,
                'validator': RTDETRValidator,
                'trainer': RTDETRTrainer,
                'model': RTDETRDetectionModel}}
