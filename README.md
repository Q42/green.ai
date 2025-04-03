# green.ai

## Installation

### Prerequisites
The project uses python managed by conda for package management and version control. To install conda, [use this tutorial](https://conda-forge.org/download/) or if using macOS use the following command:
```
brew install miniforge
```

### Installing
To install the necessary packages, run the following command.
```
conda env create -f environment.yml
```

[!NOTE] 
Not all necessary packages have been correctly updated in `enviroment.yml`. Please install missing packages via pip install <package name> and inform the repo maintainer.

Before running the code. copy the `.env.example` file into `.env` and add the openAI api key.

Find more information in the [Wiki](docs/CONTRIBUTING.md)
