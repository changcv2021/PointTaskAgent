"""Official nuScenes metric wrapper; offline evaluation only."""
from collections import Counter
from common_run import clean
def detection_metric(nusc,predictions,tokens):
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
    from nuscenes.eval.common.loaders import load_gt,add_center_dist,filter_eval_boxes
    cfg=config_factory('detection_cvpr_2019');gtall=load_gt(nusc,'val',DetectionBox,verbose=False)
    gt=EvalBoxes()
    for token in tokens:gt.add_boxes(token,gtall[token])
    pred=EvalBoxes.deserialize(predictions,DetectionBox)
    gt=filter_eval_boxes(nusc,add_center_dist(nusc,gt),cfg.class_range,verbose=False)
    pred=filter_eval_boxes(nusc,add_center_dist(nusc,pred),cfg.class_range,verbose=False)
    evaluator=NuScenesEval.__new__(NuScenesEval);evaluator.cfg=cfg;evaluator.gt_boxes=gt;evaluator.pred_boxes=pred;evaluator.verbose=True
    metric,details=evaluator.evaluate();serialized=metric.serialize()
    serialized['evaluated_gt_class_counts']=dict(Counter(b.detection_name for b in gt.all))
    return clean(serialized),clean(details.serialize())

