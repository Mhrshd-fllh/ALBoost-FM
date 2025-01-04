# ALBoost-FM
## Active Learning with YOLO and Image Blackout

### Abstract

This project implements an active learning framework for improving object detection in images using YOLO (You Only Look Once). The system uses image blackout strategies to enhance the quality of labeled data by sampling images intelligently through clustering. The approach integrates active learning with YOLOv8 to iteratively improve model accuracy by selecting the most informative samples for labeling. The project includes code for training the model and evaluating its performance.

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