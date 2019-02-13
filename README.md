# Up Your Bus Number
## A Workflow for Reproducible Data Science

**Bus Number** (bŭs nŭmʹbər), *noun*: 

*The number of people that need to get hit by a bus before your data science
project becomes irreproducible.*

This number might be **zero**. In this tutorial, we aim to increase your bus number.

## The Reproducible Data Science Process
### How do you spend your "Data Science" time?
A typical data science process involves three main kinds of tasks:
* Munge: Fetch, process data, do EDA
* Science: Train models, Predict, Transform data
* Deliver: Analyze, summarize, publish

where our time tends to be allocated something like this:

<img src="notebooks/charts/munge-supervised.png" alt="Typical Data science Process" width=500/>

Unfortunately, even though most of the work tends to be in the **munge** part of the process, when we do try and make data science reproducible, we tend to focus mainly on reprodibility of the **science** step.

That seems like a bad idea, especially if we're doing unsupervised learning, where often our time is spent like this:

<img src="notebooks/charts/munge-unsupervised.png" alt="Typical Data science Process" width=500/>

We're going to try to improve this to a process that is **reproducible from start to finish**. 

There are 4 steps to a fully reproducible data science flow:
* Creating a **Reproducible Environment**
* Creating **Reproducible Data**
* Building **Reproducible Models**
* Achieving **Reproducible Results**

In this series of tutorials, we will look at each of these steps in turn.
This repo is all about **getting you started doing Reproducible Data Science** , and giving you a **deeper look** at some of the concepts we will cover in this tutorial. For the latest version, visit: 

    https://github.com/hackalog/bus_number

To get started, open [Tutorial 1: Reproducible Environments](tutorial-1.md).

