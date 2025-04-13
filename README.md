# Image Deblurring and Denoising with UNet

This project implements an image restoration system using a UNet architecture to remove blur and noise from images. The system is trained on the DIV2K dataset and can effectively restore degraded images while preserving important details.

## Project Structure

```
.
├── data/               # Dataset directory
│   ├── raw/           # Original images
│   ├── processed/     # Processed images
│   └── splits/        # Train/val/test splits
├── models/            # Model architectures
│   └── unet.py        # UNet implementation
├── utils/             # Utility functions
│   ├── data_utils.py  # Data processing utilities
│   └── metrics.py     # Evaluation metrics
├── config.py          # Configuration parameters
├── train.py           # Training script
├── test.py            # Testing script
├── download_data.py   # Dataset download script
└── requirements.txt   # Project dependencies
```

## Features

- Image deblurring and denoising using UNet architecture
- Support for both training and inference
- Evaluation metrics: PSNR and SSIM
- TensorBoard integration for training visualization
- Checkpoint saving and loading
- Configurable parameters for blur and noise levels

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/image-restoration.git
cd image-restoration
```

2. Create and activate a virtual environment:
```bash
conda create -n image_restoration python=3.8
conda activate image_restoration
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Download the dataset:
```bash
python download_data.py
```

2. Train the model:
```bash
python train.py
```

3. Test the model:
```bash
python test.py
```

## Configuration

The `config.py` file contains all configurable parameters:

- Training parameters (batch size, epochs, learning rate)
- Model architecture (number of filters, channels)
- Image degradation parameters (blur kernel size, noise level)
- Dataset paths and splits
- Evaluation metrics

## Results

The model achieves the following performance:
- PSNR: ~25 dB
- SSIM: ~0.85

Sample results are saved in the `results` directory, showing:
- Original images
- Degraded images (with blur and noise)
- Restored images (after model processing)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DIV2K dataset for training data
- PyTorch for deep learning framework
- UNet architecture for image restoration 