# Tutorial 1: Reproducible Environments

## Overview

* Requirements: The Bare Minimum 

* Using a Data Science Template: `cookiecutter`

* Virtual Environments: `conda` and environment files
* Revision Control: git and a git workflow
   * Installing, Enabling, and using nbdime
* The Data Science DAG
   * make, Makefiles and data flow
* Python Modules
   * Creating an editable module
* Testing: doctest, pytest, hypothesis

## The Bare Minimum
You will need:
* `conda` (via anaconda or miniconda)
* `cookiecutter` 
* `make`
* `git`
* `python >= 3.6` (via `conda`)

### ASIDE: Our Favourite Python Parts
Why the `python>=3.6` requirement?
* f-strings: Finally, long, readable strings in our code.
* dictionaries: insertion order is preserved!

Other great tools:
* `pathlib`: Sane, multiplatorm path handling: https://realpython.com/python-pathlib/
* `doctest`: Examples that always work: https://docs.python.org/3/library/doctest.html
* `joblib`: Especially the persistence part: https://joblib.readthedocs.io/en/latest/persistence.html

 ### Installing Anaconda
We use `conda` for handling package dependencies, maintaining virtual environments, and installing particular version of python. For proper integration with pip, you should make sure you are running conda >= 4.4.0. Some earlier versions of conda have difficulty with editable packages (which is how we install our `src` package)

* See the [Anadonda installation guide](https://conda.io/docs/user-guide/install/index.html) for details

### Installing Cookiecutter
`cookiecutter` is a python tool for creating projects from project templates. We use cookiecutter to create a reproducible data science template for starting our data science projects.

To install it:
```
  conda install -c conda-forge cookiecutter
```
### make
We use gnu `make` (and `Makefiles`) as a convenient interface to the various stages of the reproducible data science data flow. If for some reason your system doesn't have make installed, try:
```
  conda install -c anaconda make
```
### git
We use git (in conjunction with a workflow tool like GitHub, BitBucket, or GitLab) to manage version control. 

Atlassian has good [instructions for installing git](https://www.atlassian.com/git/tutorials/install-git) if it is not already available on your platform.

### Exercise 1: Install the requirements
* Anaconda
* Cookiecutter
* make
* git

Check your installations:

```
conda --version
make --version
git --version
cookiecutter --version
```

## Using a Data Science Template: `cookiecutter`

We use cookiecutter to create a reproducible data science template for starting our data science projects.


You can install from the github/bitbucket repo directly, even from a particular branch. For example, we will want to use the `bus_number` branch of `cookiecutter-easydata`.

```
cookiecutter https://github.com/hackalog/cookiecutter-easydata.git --checkout bus_number
```

**Note**: Once you have completed this tutorial, we recommend that you use the standard version of `cookiecutter-easydata` as `bus_number` branch has been customized for this tutorial:
```
cookiecutter https://github.com/hackalog/cookiecutter-easydata.git
```

### Exercise 2: Start your cookiecutter-based project
Create a project called `Bus Number Tutorial`:
* Use `conda` as your virtualenv manager
* Use python 3.6 or greater

When complete, you should have a fully populated project directory (`bus_number_tutorial`), complete with customized `README.md`.

We will be working in this project from now on.


##  Virtual Environments: `conda` and environment files

Everyone's computing environment is different. How can we ensure that another user running a different platform can successfully run the code you are creating? How do we know they are using the same versions of your code and all its various supporting libraries? How do we reproduce your working environment on someone else's machine?

In short, by using **virtual environments**. 

In this toolkit, we use `conda` (as provided by either *anaconda* or *miniconda*) to create and manage virtual environments. Furthermore, we use an **environment file**, `environment.yml` to specify all of the dependencies that need to be installed to run our code.
    
Two `make` commands  ensure that we have the appropriate environment. Use
* `make create_environment`: for the initial creation of a project specific conda environment
* `make requirements`: to update your environment whenever you change your `environment.yml` specs.

If you ever get the urge to delete your environment and start again from scratch, you can do a
* `make delete_environment`

We will get to `make` in the next section of this tutorial.

**Caveat**: Technically speaking, a `conda` environment created from an `environment.yml` file is likely **not reproducible**. Even if you specify a specific version of a package in your `environment.yml`, the way its dependencies get resolved may differ from system to system. One way to work around this ambiguity is to have an additional file (called a **lockfile**) that explicitly records all dependencies and version numbers. This is the **great way** to handle ambiguity, while keeping your `environment.yml` requirements manageable. In this toolkig, we have implemented this lockfile mechanism by automatically generating an `environment.lock` file from your `environment.yml` whenever it changes.

### Exercise 3: Set up your virtual environment and install all dependencies
* Create and activate your `bus_number_tutorial` conda environment using the above `make` commands.
* Look at the difference between `environment.yml` and the generated `environment.lock`

### Exercise 4: Pick up this tutorial in your new conda environment
* Run `jupyter notebook` and open `notebooks/10-reproducible-environment.ipynb`.
