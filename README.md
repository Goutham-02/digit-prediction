# Digit Prediction

This repository contains a Python project for digit recognition using a neural network. The project provides scripts for model creation, training, inference, and visualization (using Grad-CAM), as well as a simple web interface for testing the model.

## Features

- Digit recognition using a neural network.
- Pre-trained model (`model.pth`) included.
- Grad-CAM visualization to interpret model predictions.
- Web interface (`index.html`) for easy digit input and prediction.
- Utility scripts for model training, inference, and visualization.

## Repository Structure

- `digit.py`: Main script for digit recognition.
- `main.py`: Entry point for running the application.
- `model.py`: Model architecture definition.
- `model.pth`: Pre-trained model weights.
- `gradcam.py`: Grad-CAM visualization implementation.
- `utils.py`: Utility functions.
- `index.html`: Web interface for digit prediction.
- `requirements.txt`: List of Python dependencies.
- `.gitignore`: Git ignore rules.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- pip

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Goutham-02/digit-prediction.git
   cd digit-prediction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Usage

#### Run the Digit Recognition Model

You can use the scripts provided for various tasks:

- **Train or evaluate the model:**  
  Run `main.py` or `digit.py` as appropriate. (See code comments for specific usage.)

- **Web Interface:**  
  Open `index.html` in your browser to draw a digit and see the prediction from the model.

#### Grad-CAM Visualization

To visualize model predictions, use the `gradcam.py` script to generate heatmaps for input digits.

## Model

The model is a neural network (defined in `model.py`) trained to recognize handwritten digits (e.g., MNIST-style images). The pre-trained weights are included as `model.pth` for immediate inference.

## Requirements

All Python dependencies are listed in `requirements.txt`. Typical requirements may include:
- torch
- torchvision
- numpy
- matplotlib
- flask (if a web server is used)

## License

This project currently does not specify a license. Please add a license if you intend to share or reuse the code.

---

**Author:** [Goutham-02](https://github.com/Goutham-02)

For questions or contributions, feel free to open an issue or pull request.
