import os
import dtlpy as dl
from mmdet.apis import init_detector, inference_detector
import logging
import mmrotate
import numpy as np

logger = logging.getLogger('MMRotate')


@dl.Package.decorators.module(description='Model Adapter for mmlabs object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class MMRotate(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        config_file = 'rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'
        checkpoint_file = 'rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth'

        if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
            logger.info("Downloading mmrotate artifacts")
            os.system("mim download mmrotate --config rotated_faster_rcnn_r50_fpn_1x_dota_le90 --dest .")

        logger.info("MMDetection artifacts downloaded successfully, Loading Model")
        self.model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'
        logger.info("Model Loaded Successfully")
        self.classes = self.model.CLASSES

    def predict(self, batch, **kwargs):
        logger.info(f"Predicting on batch of {len(batch)} images")
        batch_annotations = list()
        for image in batch:
            image_annotations = dl.AnnotationCollection()
            detections = inference_detector(self.model, image)
            if isinstance(detections, tuple):
                bbox_result, segm_result = detections
            else:
                bbox_result, segm_result = detections, None
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            bboxes = np.vstack(bbox_result)
            for bbox, label in zip(bboxes,
                                   labels):
                confidence = bbox[5]
                if confidence >= 0.4:
                    # min_x = xc
                    # min_y = yc
                    # max_x = xc + w
                    # max_y = yc + h
                    # angle = math.cos(math.radians(ag))
                    xc, yc, w, h, ag = bbox[:5]
                    wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
                    hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
                    p1 = (xc - wx - hx, yc - wy - hy)
                    p2 = (xc + wx - hx, yc + wy - hy)
                    p3 = (xc + wx + hx, yc + wy + hy)
                    p4 = (xc - wx + hx, yc - wy + hy)
                    poly = np.int0(np.array([p1, p2, p3, p4]))
                    image_annotations.add(annotation_definition=dl.Polygon(geo=poly,
                                                                           label=self.classes[label]),
                                          model_info={'name': self.model_entity.name,
                                                      'model_id': self.model_entity.id,
                                                      'confidence': confidence})
            batch_annotations.append(image_annotations)
            logger.info(f"Found {len(image_annotations)} annotations in image")
        return batch_annotations
