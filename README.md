# Matrix Inversion using Multi-Layer Perceptron (MLP)

This repository will showcase my journey of me learning how to use a Multi-Layer Perceptron (MLP) to approximate the inverse of a square matrices.

## Project Structure

```sh
matrix_inversion_MLP/
|-- data.py               # Data generation & preprocessing
|-- MLP.py                # MLP model architecture
|-- train.py              # Model training script
|-- evaluate.py           # Model evaluation script
|-- requirements.txt      # Required Python packages
|__ README.md             # Project documentation
```

## My Learning journey

**1. Starting with Data Generation**
In order to generate the data, I used numpy to generate random matrices. Later on in the project, I implemented a filter to remove near-singular matrices by checking their determinants. A near-singular matrices means that the matrice's determinant is close to 0, which essentially makes it non-invertible.

**2. Moving onto Preprocessing**
Next, I had to preprocess the data. I had originally flattened the matrices and normalized the data, but then realized that normalizing the data could distort the properties needed for inversion. So I had to remove the normalization which really helped in the long run.

**3. Building the MLP**
Designing the MLP architecture was both fun and challenging. I experimented with different numbers of layers and neurons, tested different activation functions, and eventually introduced residual connections to help the model learn better. Residual connections add the original input to the output of a layer. Adding dropout layers to prevent overfitting was really important in the progression of the model.

**4. Issues with Training**
Training the model was where I encountered the most difficulties. The loss plateaued early, and the model failed to generalize often. This led me to:

- Implement learning rate scheduling
- Eventually removed this and stuck with 0.0003
- Apply gradient clipping to prevent gradients
- Used to fix gradients becoming exceedingly large during training
- Use early stopping to quit training when improvements stalled
- To stop the training when loss function plateaued

## Installation

1. Clone the repository

```sh
git clone https://github.com/anishbanda/matrix_inversion_MLP
cd matrix_inversion_MLP
```

2. Set up a virtual environment (recommended)

```sh
python3 -m venv venv
```

3. Install dependencies

```sh
pip install -r requirements.txt
```

## Running the Code

**Training the Model:**

```sh
python3 train.py
```

Trains the MLP. Watch as the loss decreases (the closer to 0, the better).

**Evaluating the Model:**

```sh
python3 evaluate.py
```

This compares the model's predicted inverses with actual inverses. Won't be 100% unless the loss function turns into 0.

## Challenges I Faced

**1. Overfitting**

- Added dropout layers and used early stopping

**2. Vanishing/Exploding Gradients**

- Implemented gradient clipping

**3. Loss Plateau**

- Adjusted learning rates and introduced learning rate scheduling

**4. Data Issues**

- Filtered out nearly singular matrices and removed data normalization

## Key Takeaways

- It was all about iterating. Nothing worked perfectly, I just had to keep trying, until I eventually got close.
- Preprocessing decisions significantly impacted model performance.
- Neural networks take a while to learn, patience is key.
