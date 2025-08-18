# AI-powered System for Early Detection of Nail Diseases Using CNNs and Image Processing

## Overview

This repository contains an AI-powered system designed to assist in the early detection of nail diseases using Convolutional Neural Networks (CNNs) and advanced image processing techniques. The system aims to provide accurate and efficient diagnosis by analyzing nail images, potentially supporting dermatologists and healthcare professionals.

## Features

- **Image Preprocessing:** Techniques such as resizing, normalization, and augmentation to improve model accuracy.
- **CNN Architecture:** Deep learning models trained to classify various nail diseases.
- **Disease Classification:** Detects and categorizes common nail diseases from input images.
- **Performance Metrics:** Reports accuracy, precision, recall, and F1-score.
- **User Interface:** Easy-to-use interface for uploading and analyzing nail images (optional/coming soon).

## Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/gl1461/AI-powered-system-for-early-detection-of-nail-diseases-using-CNNs-and-image-processing.git
    cd AI-powered-system-for-early-detection-of-nail-diseases-using-CNNs-and-image-processing
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download datasets:**
    - Place your nail disease image dataset in the `data/` directory.

## Usage

1. **Train the model:**
    ```bash
    python train.py --data_dir data/
    ```

2. **Evaluate the model:**
    ```bash
    python evaluate.py --model_path models/best_model.pth --data_dir data/test/
    ```

3. **Predict on new images:**
    ```bash
    python predict.py --image sample_nail.jpg
    ```

## Project Structure

```
├── data/                # Dataset directory
├── models/              # Saved models
├── src/                 # Source code
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── predict.py
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
```

## Model Details

- **Architecture:** Custom CNN (details in `src/model.py`)
- **Input:** Nail image (JPEG, PNG)
- **Output:** Predicted disease class

## Dataset

Use publicly available or curated datasets of nail images with labeled diseases. Ensure images are high-quality and properly annotated.

## Contributing

Contributions are welcome! Please submit pull requests or open issues for suggestions, bug reports, or feature requests.

## License

This project is licensed under the MIT License.

## Contact

For questions or collaborations, please contact [gl1461](https://github.com/gl1461).
