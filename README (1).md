# Fresh vs. Spoiled Fruit Classification

An image classification project that uses **Vision Transformer (ViT)** to distinguish between fresh and spoiled fruits. The project explores transfer learning and fine-tuning strategies using TensorFlow/Keras and PyTorch with Hugging Face Transformers.

## Project Overview

The goal is to classify fruit images into two categories:

- **Fresh**
- **Spoiled**

The notebook experiments with pretrained Vision Transformer models and compares different training strategies, including a frozen feature extractor, partial fine-tuning, and full fine-tuning.

> Note: The dataset directory in the notebook uses the class folder name `Frish` (rather than `Fresh`). The README uses the intended label **Fresh** for clarity.

## Dataset

The notebook loads the image dataset from a ZIP file stored in Google Drive and extracts it in Google Colab.

- **Number of images:** 10,000
- **Number of classes:** 2
- **Classes:** `Frish` (Fresh) and `Spoiled`
- **Image formats supported by the loader:** JPG, JPEG, PNG
- **Image size used by the model:** 224 × 224 pixels

The images are organized into class folders, with fruit-specific subfolders inside each class directory.

### Data Split

The adopted PyTorch implementation uses stratified splitting to preserve the class distribution:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 7,000 | 70% |
| Validation | 1,500 | 15% |
| Test | 1,500 | 15% |
| **Total** | **10,000** | **100%** |

The split uses `random_state=42`.

## Project Workflow

### 1. Data Loading and Preparation

- Load and extract the dataset.
- Collect image paths and labels from the class folders.
- Split the data into training, validation, and test sets using stratified sampling.
- Resize images to 224 × 224 pixels.
- Apply random horizontal flipping to training images.

### 2. Model Experiments

Two implementation approaches were explored.

#### Approach 1: TensorFlow / Keras with TensorFlow Hub ViT

- Pretrained ViT-B16 feature extractor from TensorFlow Hub: [`vit_b16_fe`](https://tfhub.dev/sayakpaul/vit_b16_fe/1).
- Data augmentation using random horizontal flipping, rotation, and zoom.
- A Dropout layer followed by a Dense Softmax classification head.
- Initial training with the ViT feature extractor frozen, followed by an attempted fine-tuning stage.

Unfreezing the full TensorFlow Hub model caused the Colab session to run out of available RAM. This approach was therefore not selected as the final implementation.

#### Approach 2: PyTorch with Hugging Face ViT (Adopted Approach)

The adopted implementation uses the pretrained model:

[`google/vit-base-patch16-224-in21k`](https://huggingface.co/google/vit-base-patch16-224-in21k)

The model is adapted for binary classification with two output labels.

**Training configuration:**

| Parameter | Value |
|---|---|
| Input image size | 224 × 224 |
| Batch size | 16 |
| Optimizer | AdamW |
| Loss function | CrossEntropyLoss |
| Initial learning rate | 5e-5 |
| Random seed | 42 |

**Training strategy:**

1. **Frozen backbone:** Train the classification head while keeping the pretrained ViT backbone frozen for 5 epochs.
2. **Partial fine-tuning:** Unfreeze the last encoder block and train for 3 additional epochs with a learning rate of 1e-5.
3. **Full fine-tuning:** Unfreeze all ViT layers and train for 3 additional epochs with a learning rate of 5e-6.

This progressive fine-tuning strategy allows the model to adapt its pretrained visual representations to the fruit classification task.

## Results

The notebook reports the following test accuracy at different training stages:

| Training Stage | Reported Test Accuracy |
|---|---:|
| Frozen ViT backbone | 96.58% |
| Last encoder block unfrozen | 99.75% |
| Full fine-tuning | **100%** |

The final evaluation output in the notebook also reports:

| Metric | Reported Score |
|---|---:|
| Accuracy | 100% |
| Precision (weighted) | 100% |
| Recall (weighted) | 100% |
| F1-score (weighted) | 100% |

**Evaluation note:** The notebook's split reports 1,500 test images, while the final classification report lists 1,200 test samples. The reported 100% metrics should therefore be verified by rerunning the final evaluation and confirming that the same complete test set is used throughout. These values are reported as shown in the notebook, not as independently verified results.

## Prediction

The notebook includes a prediction function that:

1. Loads an image from a given path.
2. Converts it to RGB and resizes it to 224 × 224.
3. Extracts pixel values using the pretrained ViT feature extractor.
4. Passes the image through the trained model.
5. Returns the predicted class label.

## Technologies & Libraries

- Python
- PyTorch
- TensorFlow / Keras
- Hugging Face Transformers
- TensorFlow Hub
- torchvision
- scikit-learn
- NumPy
- Pillow (PIL)
- tqdm
- Google Colab

## Repository Structure

```text
.
├── FRESHVS.SPOILEDFRUITCLASSIFICATION.ipynb
├── README.md
└── requirements.txt
```

The notebook contains dataset preparation, model experiments, training, evaluation, and image prediction.

## How to Run

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

3. Upload the fruit dataset ZIP file to Google Drive, or update the dataset path in the notebook to match your local dataset directory.

4. Open the notebook in Google Colab or Jupyter Notebook.

5. Run the cells in order. A GPU runtime is recommended for ViT training and fine-tuning.

## Notes

- The dataset is not included in this repository. You must provide the dataset separately.
- The notebook was developed using Google Colab and Google Drive paths; update paths when running in another environment.
- Training time and memory requirements depend on the available hardware.
- This project is an educational computer vision experiment and is not intended for food safety certification or automated commercial quality control.

## Contributors

- **Shahed Ayham Abosaeed**
