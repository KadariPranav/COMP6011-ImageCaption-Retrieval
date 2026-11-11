# Data and reproducibility notes

## Source and scope

COCO 2017 validation subset supplied as the group's shared archive `multimodal_shared_data(1).zip` (5,000 images and 25,010 captions). This work uses **precomputed** ResNet-50 (2048D) and BERT (768D) features. The feature-extraction stage is a shared input, not individually rerun in this notebook. See the assignment instructions for the original COCO source (`lmms-lab/COCO-Caption2017`, split `val`).

The ZIP contains:

```
shared_data/coco_splits.json
shared_data/image_features.pt
shared_data/text_features.pt
results/dataset_audit.json
```

The notebook asserts image ID order and feature dimensions, checks caption→image indices and split membership, and confirms numerical finiteness. It forms image-disjoint train/validation/test splits of 4,000/500/500 images (20,008/2,499/2,503 captions).

## Re-run

1. Download the original shared feature ZIP through the course-approved location (it is deliberately **not included in GitHub**).
2. Place `multimodal_shared_data(1).zip` next to the notebook or upload it into the Colab working directory.
3. Install the dependencies in `requirements.txt` (select an appropriate PyTorch build for your device).
4. Run every notebook cell in order from a clean runtime. The archive is extracted automatically, and `individual_coco_results/` is generated locally.
5. Check the saved metrics and plots. CPU/GPU variation can change run times and, depending on backend determinism, model metrics. The archived CSVs record results from the supplied executed notebook, not promised rerun values.

## Evaluation protocol

Models: symmetric InfoNCE (8 epochs); top-8 triplet ranking (up to 24 epochs; patience 6, selected checkpoint epoch 16); early fusion pair classifier (8 epochs). Seed 42. Model selection uses the validation split; test retrieval is reported for image-to-text and text-to-image Recall@1, @5, @10, and MRR.

## Limitations

One random seed; pretrained frozen feature encoders; dataset/subgroup bias untested. Bootstrapped confidence intervals reflect query resampling, not training-run variance. Encoder extraction time is excluded from runtime comparisons. Error categories require visual inspection to confirm.
