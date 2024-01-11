# ProjectX-Coevolutionary-Selection-Training

This repository contains our submission for the University of Toronto's ProjectX - Theme: Efficient AI. We use a coevolutionary algorithm to train a deep neural network classifier on various image datasets, such as CIFAR10, CIFAR100, and ImageNet. We also compare the performance of our method with and without the use of a prey population, which acts as a dynamic and adaptive fitness function for the predator (classifier).

## Requirements

To run this project, you need to have the following dependencies installed:

- Python 3.8 or higher
- PyTorch 1.9 or higher
- DEAP 1.3 or higher
- Wandb 0.12 or higher
- Matplotlib 3.4 or higher
- Numpy 1.21 or higher

You also need to create a .env file in the root directory of the project and add the following variables:

- CIFAR100_DIR={Path to a CIFAR100 file}
- CIFAR10_DIR={Path to CIFAR10 file}
- ImageNet_DIR={Path to ImageNet file}

## Usage

To run the coevolutionary algorithm with the predator and prey, use the following command:

`python devMain.py`

This will create a wandb project and log the results of the evolution, such as the fitness, accuracy, and diversity of the populations. You can also use the `better_logging_graphs.py` script to generate graphs for future comparisons.

To run the coevolutionary algorithm without the prey, use the following command:

`python mainCompareMain.py`

This will also create a wandb project and log the results of the evolution, similar to the previous script. You can compare the performance of the two methods by using the wandb dashboard or the `better_logging_graphs.py` script.

## Resources

For more information about the coevolutionary algorithm and the competitive learning scheme, you can refer to the following papers:

- Competitive Learning Scheme for DNN classifier training
- A Survey of Coevolutionary Algorithms

You can also check out the DEAP documentation for more details about the evolutionary framework:

- DEAP docs

## Credits

This project is based on the following open-source repositories:

- DEAP
- gitignore template
