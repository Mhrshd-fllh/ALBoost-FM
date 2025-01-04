# ALBoost-FM
## Enhancing Active Learning Performance with Initial Data Selection Based on Foundation Models

### Abstract

In active object detection, selecting the initial labeled
pool is very important, especially in complex datasets with
expensive labeling and low resource budget. This study
presents an active learning framework designed to optimize
the initial training cycle of active learning for the
object detection task. Our approach diverges from traditional
methods by utilizing feature-based clustering from
the YOLO-World model and refining images based on objectness
scores from YOLO rather than relying on uncertainty
sampling in the initial cycle. This approach focuses
on finding samples with various features and emphasizes
training in areas where the foundation model is most confident.
Doing so helps the model better understand different
types of objects. To evaluate how well this method
works, we use two complex datasets called Aquarium and
NWPU VHR-10, which are well known for their diverse object
sizes and challenging conditions. Subsequent cycles
employ a consistency-based sampling technique, maintaining
the robustness of the training process. In the NWPU
dataset, our clustering method increases mAP50 by an average
of 6.6% in image-based evaluation compared to random
sampling. Moreover, our clustering with the image refinement
method shows an improvement of approximately
11% in object-based evaluation. The results show improvements
in model performance and labeling time, demonstrating
the advantages of using foundation models in the early
cycles of active learning.
---

## Plot Space

> Add any relevant plot or visualizations here. This can be a sample output or any plots that help explain the project's process or results.

---

## Guide for Training

Once the environment is set up and the model is ready, you can start training the YOLOv8 model using the following steps.

### Step 1: Run the Active Learning Script

This script trains the YOLOv8 model using the active learning process, iteratively improving the model over multiple cycles.

```bash
python active_learning.py \
    --cycles 9 \
    --num_samples 50 \
    --num_samples2 10 \
    --num_clusters 20 \
    --model_path yolov8x-worldv2.pt \
    --train_epochs 20 \
    --output_image_dir /content/datasets/VOC_l/images/train2012 \
    --output_label_dir /content/datasets/VOC_l/labels/train2012 \
    --weight1 0.95 \
    --weight2 0.05
```

### Explanation:

- `--cycles`: Number of active learning cycles to run.
- `--num_samples`: Number of images to sample in the first cycle.
- `--num_samples2`: Number of images to sample in subsequent cycles.
- `--num_clusters`: Number of clusters for KMeans.
- `--model_path`: Path to the pre-trained YOLO model.
- `--train_epochs`: Number of epochs to train the model.
- `--output_image_dir`: Directory to save output images.
- `--output_label_dir`: Directory to save output labels.
- `--weight1`: Weight for image uniformity.
- `--weight2`: Weight for pool uniformity.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.