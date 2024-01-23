import os
import dtlpy as dl
from mmdet.apis import init_detector, inference_detector
import logging
import mmrotate
import numpy as np
import torch
import subprocess

logger = logging.getLogger('MMRotate')


@dl.Package.decorators.module(description='Model Adapter for mmlabs object detection',
                              name='model-adapter',
                              init_inputs={'model_entity': dl.Model})
class MMRotate(dl.BaseModelAdapter):
    def load(self, local_path, **kwargs):
        model_name = self.model_entity.configuration.get('model_name',
                                                         'rotated_faster_rcnn_r50_fpn_1x_dota_le90')
        config_file = self.model_entity.configuration.get('config_file',
                                                          'rotated_faster_rcnn_r50_fpn_1x_dota_le90.py')
        checkpoint_file = self.model_entity.configuration.get('checkpoint_file',
                                                              'rotated_faster_rcnn_r50_fpn_1x_dota_le90-0393aa5c.pth')

        if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
            logger.info("Downloading mmrotate artifacts")
            download_status = subprocess.Popen(f"mim download mmrotate --config {model_name} --dest .",
                                               stdout=subprocess.PIPE,
                                               stderr=subprocess.PIPE,
                                               shell=True)
            download_status.wait()
            if download_status.returncode != 0:
                (out, err) = download_status.communicate()
                raise Exception(f'Failed to download mmrotate artifacts: {err}')

        logger.info("MMDetection artifacts downloaded successfully, Loading Model")
        device = self.model_entity.configuration.get('device', 'cuda:0')
        if device == 'cuda:0':
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.confidence_thr = self.model_entity.configuration.get('confidence_thr', 0.4)
        self.model = init_detector(config_file, checkpoint_file, device=device)  # or device='cuda:0'
        logger.info("Model Loaded Successfully")

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
                np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)
            ]
            labels = np.concatenate(labels)
            bboxes = np.vstack(bbox_result)
            for bbox, label in zip(bboxes,
                                   labels):
                confidence = bbox[5]
                if confidence >= self.confidence_thr:
                    xc, yc, w, h, ag = bbox[:5]
                    wx, wy = w / 2 * np.cos(ag), w / 2 * np.sin(ag)
                    hx, hy = -h / 2 * np.sin(ag), h / 2 * np.cos(ag)
                    p1 = (xc - wx - hx, yc - wy - hy)
                    p2 = (xc + wx - hx, yc + wy - hy)
                    p3 = (xc + wx + hx, yc + wy + hy)
                    p4 = (xc - wx + hx, yc - wy + hy)
                    poly = np.int0(np.array([p1, p2, p3, p4]))
                    image_annotations.add(annotation_definition=dl.Polygon(geo=poly,
                                                                           label=self.model_entity.labels[label]),
                                          model_info={'name': self.model_entity.name,
                                                      'model_id': self.model_entity.id,
                                                      'confidence': confidence})
            batch_annotations.append(image_annotations)
            logger.info(f"Found {len(image_annotations)} annotations in image")
        return batch_annotations
