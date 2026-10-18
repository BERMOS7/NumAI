# NumAI

NumAI is a desktop application for recognizing handwritten digits. It combines a
Tkinter drawing interface with a fully connected neural network implemented from
scratch with NumPy.

The project demonstrates the main steps of neural-network training: forward
propagation, activation functions, loss calculation, gradient computation,
backpropagation, and stochastic gradient descent.

## Features

- Draw a digit with the mouse and get a prediction from the desktop interface.
- Preprocess drawings into 28 × 28 pixel images.
- Load and normalize MNIST data from compressed IDX archives.
- Train the neural network and save its parameters as JSON.
- Use the included pretrained model for predictions without downloading MNIST.
- Run unit tests with Python's built-in `unittest` framework.

## Neural network

NumAI uses a multilayer perceptron (MLP), a fully connected neural network in
which each neuron in a layer connects to every neuron in the next layer.

The current architecture is:

```text
Input: 784 values (28 × 28 pixels)
       │
       ▼
Hidden layer 1: 64 neurons
       │ ReLU
       ▼
Hidden layer 2: 64 neurons
       │ ReLU
       ▼
Output layer: 10 neurons
       │ Softmax
       ▼
Probabilities for digits 0 through 9
```

The layer sizes are configured as `LAYERS = [64, 64, 10]`; the 784 input pixels
are not included in this list. The network has 55,050 trainable parameters:

| Connection | Weights | Biases | Total |
|---|---:|---:|---:|
| Input → hidden layer 1 | 50,176 | 64 | 50,240 |
| Hidden layer 1 → hidden layer 2 | 4,096 | 64 | 4,160 |
| Hidden layer 2 → output | 640 | 10 | 650 |
| **Total** | **54,912** | **138** | **55,050** |

Hidden layers use ReLU. The output layer uses Softmax to convert its ten scores
into probabilities that sum to one. Cross-entropy measures the difference
between these probabilities and the expected label. The model is trained with
stochastic gradient descent (SGD), updating its parameters after each example.

## MNIST data

MNIST contains 60,000 training images and 10,000 test images. Each is a grayscale
28 × 28 image of a handwritten digit from 0 to 9. The data files are not included
in this repository; the pretrained model is included, so the drawing application
can still be used without the dataset.

To train the model, download these four compressed IDX files from
[Google's MNIST hosting](https://storage.googleapis.com/cvdf-datasets/mnist/)
and place them in the `MNIST/` directory at the project root:

```text
MNIST/
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
├── t10k-images-idx3-ubyte.gz
└── t10k-labels-idx1-ubyte.gz
```

The application reads the `.gz` files directly; there is no need to extract
them. The training set updates the network parameters, while the test set is
used to measure accuracy on examples that are not used for those updates.

## Requirements

- Python 3.10 or later.
- NumPy and Pillow, installed from `requirements.txt`.
- Tkinter, usually included with Python. Some Linux distributions package it
  separately (for example, `python3-tk`).

## Installation

From the repository root, create and activate a virtual environment.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### macOS and Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the application

From the repository root:

```bash
python main.py
```

Draw a digit and click **Predict**. Click **Clear** to start over. The pretrained
model is loaded from `model_NumAI/model.json`.

## Train the model

After placing the four MNIST archives in `MNIST/`, run:

```bash
python -m model_NumAI.train
```

The training script converts each label to a one-hot vector and trains the
`784 → 64 → 64 → 10` network for ten epochs, using a learning rate of `0.01`.
It evaluates accuracy on the test set after each epoch and saves the model to
`model_NumAI/model.json` at the end of every epoch.

**Warning:** training overwrites the existing model file. Keep a backup if you
want to preserve the pretrained model.

## Tests

Run the unit tests from the project root:

```bash
python -m unittest discover -s tests -v
```

The tests cover MNIST file loading, drawing preprocessing, and model prediction.

## Project structure

```text
.
├── main.py                 # Application entry point
├── UI/                     # Interface, drawing, and image preprocessing
├── model_NumAI/
│   ├── network.py          # Neural network and training algorithm
│   ├── train.py            # Dataset loading and model training
│   ├── mnist.py            # MNIST IDX file loading
│   ├── predict.py          # Model loading and digit prediction
│   └── model.json          # Pretrained model
├── MNIST/                  # Local dataset files (not tracked by Git)
├── tests/                  # Unit tests
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Technologies

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Matrix operations and numerical calculations |
| Tkinter | Desktop user interface |
| Pillow | Image processing |
| MNIST | Training and test data |
| JSON | Model storage |
| `unittest` | Automated tests |

The learning algorithm is implemented with NumPy and does not use a specialized
deep-learning framework such as PyTorch or TensorFlow.

## Limitations and possible improvements

NumAI currently classifies individual handwritten digits. Its fully connected
architecture does not exploit image structure as specialized convolutional
neural networks do. Possible improvements include plotting training loss,
showing probabilities for every digit, experimenting with architectures and
learning rates, and comparing the MLP with a convolutional network.

## License

No license has been added yet. The code is currently published without an
explicit open-source license, so reuse, modification, and redistribution are
not expressly granted.
