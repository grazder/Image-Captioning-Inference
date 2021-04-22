from torchvision import transforms as trn
import numpy as np
import torch
from ml import views

NUM_OBJECTS = 100

from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, \
    fast_rcnn_inference_single_image


class ImagePreprocessing:

    def __init__(self, embed_type: str) -> None:
        if embed_type == 'resnet' or embed_type == 'bottom-up':
            self.embed_type = embed_type
        else:
            raise ValueError('Wrong Preprocessing Type')

    def preprocess(self, img: str):
        if self.embed_type == 'resnet':
            return self._preprocess_resnet(img)
        elif self.embed_type == 'bottom-up':
            return self._preprocess_bottom_up(img)

    def _preprocess_resnet(self, img):
        preprocess = trn.Compose([
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)

        img = img[:, :, :3].astype('float32') / 255.0

        img = torch.from_numpy(img.transpose([2, 0, 1]))
        img = preprocess(img)
        with torch.no_grad():
            tmp_fc, tmp_att = views.MY_RESNET(img)

        return tmp_fc, tmp_att.reshape(1, -1, 2048)

        # fc_batch[0] = tmp_fc.data.cpu().float().numpy()
        # att_batch[0] = tmp_att.data.cpu().float().numpy()
        #
        # return fc_batch, att_batch.reshape(1, -1, 2048)

    def _preprocess_bottom_up(self, img):
        predictor = views.MY_BOTTOM_UP

        with torch.no_grad():
            raw_height, raw_width = img.shape[:2]

            # Preprocessing
            image = predictor.transform_gen.get_transform(img).apply_image(img)

            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = [{"image": image, "height": raw_height, "width": raw_width}]
            images = predictor.model.preprocess_image(inputs)

            # Run Backbone Res1-Res4
            features = predictor.model.backbone(images.tensor)

            # Generate proposals with RPN
            proposals, _ = predictor.model.proposal_generator(images, features, None)
            proposal = proposals[0]

            # Run RoI head for each proposal (RoI Pooling + Res5)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            features = [features[f] for f in predictor.model.roi_heads.in_features]
            box_features = predictor.model.roi_heads._shared_roi_transform(
                features, proposal_boxes
            )
            feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1

            # Predict classes and boxes for each proposal.
            pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
            outputs = FastRCNNOutputs(
                predictor.model.roi_heads.box2box_transform,
                pred_class_logits,
                pred_proposal_deltas,
                proposals,
                predictor.model.roi_heads.smooth_l1_beta,
            )
            probs = outputs.predict_probs()[0]
            boxes = outputs.predict_boxes()[0]

            for nms_thresh in np.arange(0.5, 1.0, 0.1):
                instances, ids = fast_rcnn_inference_single_image(
                    boxes, probs, image.shape[1:],
                    score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
                )
                if len(ids) == NUM_OBJECTS:
                    break

            # instances = detector_postprocess(instances, raw_height, raw_width)
            roi_features = feature_pooled[ids].detach()

            roi_features = roi_features.reshape(1, roi_features.shape[0], 2048)

            return roi_features.mean(1), roi_features
