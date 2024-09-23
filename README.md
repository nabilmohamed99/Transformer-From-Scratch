# Transformer from scratch

This documentation covers a Transformer model implemented in PyTorch for the task of text translation. The model utilizes input embeddings, positional encoding, multi-head attention blocks, and feedforward blocks, along with functions for training.

## Model Configuration

### Function: `get_config()`
Returns a configuration dictionary containing the model's hyperparameters.

## Weight File Management

### Function: `get_weights_file_path(config, epoch: str)`
Generates the path for a weight file based on the configuration and training iteration.

## Model Classes

### 1. Class `InputEmbedding`
Generates input embeddings for tokens.

### 2. Class `PositionEncoding`
Provides positional encoding to include information about word positions.

### 3. Class `LayerNormalization`
Implements layer normalization to stabilize learning.

### 4. Class `MultiHeadAttention`
Handles multi-head attention to capture relationships between words.

### 5. Class `FeedForward`
Implements the feedforward neural network to transform representations.

### 6. Class `TransformerBlock`
Combines attention and feedforward layers.

### 7. Class `Transformer`
Implements the complete Transformer model.

## Training

### Function: `train()`
Trains the model using input data and targets.

## Evaluation

### Function: `evaluate()`
Evaluates the model on validation data.

## Running the Model

To run the model, simply import the necessary classes and functions and follow the training process.
