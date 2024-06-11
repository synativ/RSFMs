from rsfms.tasks.classification_tasks import ClassificationTask
from rsfms.tasks.multilabel_classification_tasks import MultiLabelClassificationTask
from rsfms.tasks.regression_tasks import PixelwiseRegressionTask
from rsfms.tasks.segmentation_tasks import SemanticSegmentationTask


__all__ = (
    "BATCH_IDX_FOR_VALIDATION_PLOTTING",
    "ClassificationTask",
    "MultiLabelClassificationTask",
    "PixelwiseRegressionTask",
    "SemanticSegmentationTask",
)
