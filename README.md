# lightweight-ResNet
This project will develop a lightweight ResNet with less that 5 million parameters for CIFAR-10 image classification.

# Training

The Python script `main.py` is designed to run an experiment to train ResNet on CIFAR-10.
```
python main.py [--dev]
```

## Experiment Logging
The script logs each experiment by creating a new directory in `logs` with a unique experiment id.
Each log will contain the following files:
- `logs.json`: Log file containing loss and accuracy of training and testing of each epoch.
- `ckpt.pth`: Model checkpoint file containing the model weights based on the best testing accuracy.
- `main.py`: A copy of `main.py` to log training settings and ensure reproducibility.

## Development Mode
The script can be run in development mode by passing the --dev argument when running the script.
When in development mode, only the first 100 training and testing data will be used and 1 epoch will be run.
The log will be saved in `logs/dev`.


# Inference

The Python script `inference.py` is designed to predict the test dataset.
1. Download `cifar_test_nolabels.pkl` from https://www.kaggle.com/competitions/deep-learning-mini-project-spring-24-nyu/data to `data/`.
2. Run the script with experiment id. Use multiple experiments ids for average ensemble.
    ```
    python inference.py --id <experiment_id>
    ```
3. Predictions will be saved as `predictions.csv`.

# Plots

We provide a few script for plot visualizations.

1. `analyze.py` is used to plot loss and accuracy of training and testing against epoch.
    ```
    python analyze.py --id <experiment_id>
    ```

2. `plot_parameters` is used to plot testing accuracy againt number of parameters.
    ```
    python plot_parameters.py
    ```

# References
[1] https://github.com/kuangliu/pytorch-cifar

[2] https://github.com/Nikunj-Gupta/Efficient_ResNets

[3] https://myrtle.ai/learn/how-to-train-your-resnet-5-hyperparameters/
