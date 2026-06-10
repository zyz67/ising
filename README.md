# Ising Model ML

This project uses Monte Carlo (Metropolis) simulation to generate 2D Ising model spin configurations at various temperatures and trains a Convolutional Neural Network (CNN) to distinguish between the ordered and disordered phases, identifying the critical temperature.

## Requirements
```bash
pip install numpy torch matplotlib scikit-learn seaborn
```

## Running the Code
```bash
python ising_ml.py
```

## Results
- `confusion_matrix.png`: Shows the confusion matrix of the classification.
- `accuracy_vs_T.png`: Shows the accuracy and predicted probability of the ordered phase as a function of temperature, comparing the estimated critical temperature $T_c$ with the analytical value $T_c \approx 2.269$.
- `learned_features.png`: Visualizes the learned convolutional filters (order parameter features).
