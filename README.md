# InaraNet

InaraNet is a combination Involutional Neural Network (INN) and Long Short-term Memory (LSTM) sequential deep neural network for AMP predictions.

## Installation

Obtain a copy of this repository:

```bash
$ git clone https://github.com/inaralalani/inn-lstm-amp
```

As this code uses the legacy Tensorflow 1.x, you will need to create a Python 3.7.x virtual environment. We recommend using Anaconda to manage environments and dependencies.

Install the required dependencies from the requirements file:

```bash
$ conda create -n tf1 python=3.7
$ conda activate tf1
$ pip install -r requirements.txt
```

## Usage

Inference:

```bash
$ python3 src/inference.py input/AMP.eval.fa models/amp_model_cnn.h5
```

Training:

Instructions to be added in a future update

## License

InaraNet is currently licensed under the [GNU GPLv3](https://github.com/inaralalani/inn-lstm-amp/LICENSE) license, which is subject to change.

Original work:

Daniel Veltri, Uday Kamath, and Amarda Shehu (2018) Deep Learning Improves Antimicrobial Peptide Recognition. Bioinformatics, 34(16):2740-2747. ([DOI: 10.1093/bioinformatics/bty179](https://doi.org/10.1093/bioinformatics/bty179))