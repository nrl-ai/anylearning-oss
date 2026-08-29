# Model licences

AnyLearning ships with pretrained weights so that training and auto-labelling
work on a machine that has never been online. Weights are licensed separately
from the code that loads them, so they are listed separately here.

Every model below is under a permissive licence that allows commercial use.
Nothing in this list restricts what you may do with a model you train.

## What ships with the application

| Used for | Weights | Licence |
|---|---|---|
| Object detection | NanoDet ShuffleNetV2 backbones | Apache 2.0 |
| Object detection | RF-DETR Nano and Small (COCO) | Apache 2.0 |
| Image classification | torchvision ResNet-18 / 34 / 50 (ImageNet) | BSD 3-Clause |
| Image segmentation | DeepLabV3 ResNet encoders (ImageNet) | BSD 3-Clause |
| Instance segmentation | detectron2 Mask R-CNN R50-FPN and R101-FPN | Apache 2.0 |
| Instance segmentation | RF-DETR Seg Nano and Small (COCO) | Apache 2.0 |
| Handpose classification | MediaPipe hand landmarker | Apache 2.0 |
| Auto-labelling | MobileSAM | Apache 2.0 |
| Auto-labelling | SAM 2 Hiera-Tiny and Hiera-Small | Apache 2.0 |

## Downloaded only if you ask for them

| Used for | Weights | Licence |
|---|---|---|
| Auto-labelling | SAM 2 Hiera-Base+ and Hiera-Large | Apache 2.0 |

These two are larger and less often needed, so they are fetched the first time
you select one. Everything else in the list above is already installed.

The RF-DETR checkpoints that ship are trimmed copies of the ones Roboflow
publishes: the optimiser state a training run leaves behind is removed, because
starting a new run never reads it. The model weights are unchanged.

## What this means for the models you train

The weights above are starting points. A model you train on your own data is
yours: AnyLearning claims no rights over it, and none of the licences above ask
you to publish anything, share your data, or credit anyone in your own
application.

## One thing worth knowing

Most detection and segmentation starting weights in this industry — including
the ones here — were pretrained on the COCO dataset. COCO's annotations are
CC BY 4.0, while the photographs behind them are held under a mixture of terms
by their original photographers. Shipping models pretrained this way is
standard practice across the field, and we do it too; we would rather say so
than imply a cleaner provenance than exists.

## The code

The licences of the open-source **code** in AnyLearning — several hundred
components — are reproduced in full under **Third-party licences**, next to
this document.
