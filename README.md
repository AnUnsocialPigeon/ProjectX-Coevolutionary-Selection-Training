# DEAP-evo-example
A basic DEAP algorithm in python to figure out how it works

# How to make it work
1. Create a .env file
2. Add in the following variables:
 - CIFAR100_DIR={Path to a CIFAR100 file}
 - CIFAR10_DIR={Path to CIFAR10 file}
 - ImageNet_DIR={Path to ImageNet file}

> [!Note]
> Currently, stitching different files together is not supported

>[!Custom]
> Run `wandb login` after installing wandb to add your API key to the project

![DEAP image](./DEAP%20process%20diagram.png)

# Resources
- [DEAP docs](https://deap.readthedocs.io/en/master/)
- [Competitive Learning Scheme for DNN classifier training](https://www.sciencedirect.com/science/article/abs/pii/S1568494623006804#b24)

# Credits
- [DEAP](https://deap.readthedocs.io/en/master/overview.html)
- [gitignore template](https://github.com/github/gitignore/blob/main/Python.gitignore)