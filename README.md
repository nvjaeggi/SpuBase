# SpuBase

A Python class for computing ion sputter data based on single mineral sputter data.


See the Jupyter notebook tutorials in the package directory `docs/`.


You are encouraged to fork this repository or create your own branch to add new features. You will need to issue a pull request to be able to merge your code into the main branch.

## Installation with pip
> [!WARNING]
> If you do not create a virtual environment, *SpuBase* and its dependencies will be installed into your main python environment and can mess up your other projects. Without a virtual environment this approach fails if you use an unsupported python version (<3.10) and is therefore not recommended.

1. Clone this repository (*SpuBase*) to a local directory
2. Move to the repository folder
3. **Recommended:**
   1. Create a virtual environment by using the terminal (you can also use the terminal in your IDE of preference). This command will create a local Python environment in the `.spubase_venv` directory:
       ```
       python3 -m venv .spubase_venv
       ```
   2. activate the virtual environment
      ```
      source .spubase_venv/bin/activate
      ```
4. Install *SpuBase* using pip: 
   ```
   python3 -m pip install -e .
   ```
   
## Demonstration
#### Determine path to Jupyter Notebooks
1. To locate the example Jupyter notebooks, enter Python:
    ```
    python3
    ````
2. Once in python type: 

    ```
    import spubase
    spubase.__file__
    ```
This will report the location of the *SpuBase* package on your system, from which you can determine the path to *SpuBase/docs*. This directory contains the Jupyter notebook tutorials, which you can copy to a different location if you wish. Then, exit the Python command line using `exit()`.

#### Running Jupyter Notebooks
1. When located within the *SpuBase* location, you can access the Jupyter notebook tutorials with:
    ```
    jupyter notebook SpuBase/docs/<FILENAME>.ipynb
    ```
    with `<Filename>` being either `0_Mineral_fractions` or `1_Surface_compositions`.
    * An alternative to changing directories is to give the absolute path to the notebook you want to open instead.

2. In the Jupyter notebook window you may have to *trust* the notebook for all features to work.

# Alternative ways of installation 

## Installation into Anaconda environment
Similar to the pip installation, I recommend using Anaconda to create a Python environment with advanced functionality. 

1. [Install Anaconda](https://docs.anaconda.com/free/anaconda/install/)
2. Clone this repository (*SpuBase*) to a local directory
3. Move to the repository folder
4. Create or update a conda environment by running either (this may take a while):
    ```
    conda env create -n SpuBase -f environment.yml  
    ```
    or, if you update an already active environment use:
    ```
    conda env update -f environment.yml
    ```

5. Activate the Anaconda environment with
   ```
   conda activate SpuBase
   ```
6. Install *SpuBase* using pip 
   ```
   python3 -m pip install -e .
   ``` 


## Installation for Development (Poetry)

You do not have to use any of the suggested IDEs or the Poetry (Python packaging and dependency manager), but using them makes it easier to develop *SpuBase* as a community. If you use a different IDE, please send me your installation instructions and it will be added to this README.

To work with poetry, we want to set up a virtual Python environment in the root directory of *SpuBase*. An advantage of using a virtual environment is that it remains completely isolated from any other Python environments on your system (e.g. Conda or otherwise). You must have a Python interpreter available to build the virtual environment according to the dependency in `pyproject.toml`, which could be a native version on your machine or a version from an Anaconda environment that is currently active. You only need a Python binary, so it is not required to install any packages.

### Linux 

1. Install [Poetry](https://python-poetry.org) if you do not already have it, preferentially using [pipx](https://pypa.github.io/pipx/installation/).
   ```
   pipx install poetry
   ```
2. Clone this repository (*SpuBase*) to a local directory
3. Create a virtual environment by using the terminal (you can also use the terminal in your IDE of preference). This command will create a local Python environment in the `.spubase_venv` directory:
    ```
    python3 -m venv .spubase_venv
    ```
4. Now either install *SpuBase* into your virtual environment:
   1. activate the virtual environment
      ```
      source .spubase_venv/bin/activate
      ```
   2. Install *SpuBase* using Poetry, which will satisfy all required Python package dependencies:
       ```
       poetry install
       ```
5. or continue using an IDE:
   1. Create a poetry environment in your IDE of choice
      - In VSCode, go to *File* and *Open Folder...* and select the *SpuBase* directory
      - In PyCharm, add a new project and select the *SpuBase* directory 
   2. Add the virtual Python environment as interpreter in your IDE.
     - Open a new terminal window in VSCode and VSCode should recognise that you have a virtual environment in .spubase_venv, and load this environment automatically. 
     - PyCharm should recognize the virtual environment and the poetry `pyproject.toml` file and propose the installation. If not, manually set up a _Poetry Environment_ under _Add New Interpreter > Add Local Interpreter_. Obtain the installation path of poetry in PowerShell using  
        ```
        gcm poetry
        ```
       You should now see `(.spubase_venv)` as the prefix in the terminal prompt.


### Windows PowerShell 
1. Install Python if you do not already have it. Powershell will open the Windows store where python versions are free for download and install by typing.
	```
	python
	```
2. Install [Poetry](https://python-poetry.org) if you do not already have it, preferentially using [pipx](https://pypa.github.io/pipx/installation/).
   ```
   pipx install poetry
   ```
3. Clone this repository (*SpuBase*) to a local directory
4. Create a poetry environment in your IDE of choice
   - In VSCode, go to *File* and *Open Folder...* and select the *SpuBase* directory
   - In PyCharm, add a new project and select the *SpuBase* directory 
5. Create a virtual environment by using the terminal (you can also use the terminal in your IDE of preference). This command will create a local Python environment in the `.spubase_venv` directory:
    ```
    python -m venv .spubase_venv
    ```
6. Add the virtual Python environment as interpreter in your IDE.
   - Open a new terminal window in VSCode and VSCode should recognise that you have a virtual environment in .spubase_venv, and load this environment automatically. 
   - PyCharm should recognize the virtual environment and the poetry `pyproject.toml` file and propose the installation. If not, manually set up a _Poetry Environment_ under _Add New Interpreter > Add Local Interpreter_. Obtain the installation path of poetry in PowerShell using  
	  ```
	  gcm poetry
	  ```
   You should now see `(.spubase_venv)` as the prefix in the terminal prompt.

7. If you are required to run Poetry manually in the IDE command prompt, do so with:
    ```
    poetry install
    ```

### MAC installation

1. Install [VSCode](https://code.visualstudio.com) if you do not already have it.
2. Install [Poetry](https://python-poetry.org) if you do not already have it.
3. Clone this repository (*SpuBase*) to a local directory
4. In VSCode, go to *File* and *Open Folder...* and select the *SpuBase* directory
5. You can create a virtual environment by using the terminal in VSCode, where you may need to update `python` to reflect the location of the Python binary file. This will create a local Python environment in the `.spubase_venv` directory:
	
    ```
    python -m venv .spubase_venv
    ```
6. Open a new terminal window in VSCode and VSCode should recognise that you have a virtual environment in .spubase_venv, and load this environment automatically. You should see `(.spubase_venv)` as the prefix in the terminal prompt.
7. Install the project using poetry to install all the required Python package dependencies:

    ```
    poetry install
    ```

### Common Errors

> ModuleNotFoundError: No module named ‘setuptools’

Try installing setuptools for pip (even if setuptools is found for python3)
   ```
   pip install --user pip setuptools wheel
   ```
[//]: # (To ensure that all developers are using the same settings for linting and formatting &#40;e.g., using pylint, black, isort, as installed as extensions in step 2&#41; there is a `settings.json` file in the `.vscode` directory. These settings will take precedence over your user settings for this project only.)

[//]: # (1. In VSCode you are recommended to install the following extensions:)

[//]: # (	- Black Formatter)

[//]: # (	- Code Spell Checker)

[//]: # ( 	- IntelliCode)

[//]: # (	- isort)

[//]: # (	- Jupyter)

[//]: # (	- Pylance)

[//]: # (	- Pylint)

[//]: # (	- Region Viewer)

[//]: # (	- Todo Tree)
