# SML_24025017

This repository reproduces the experiments from the paper *"Why Do Deep Convolutional Networks Generalize So Poorly to Small Image Transformations?"* by Aharon Azulay and Yair Weiss (2019).

## Installation and Running Experiments

### Setting Up the Environment
To set up the required environment, run the following commands:
```bash
conda create -n env python=3.9
pip install -r requirements.txt
```

### Running the Experiment
To reproduce the results from Table 1, execute:
```bash
python run_exp1_table1.py
```


## Results
### Table 1
![Table 1 Results](assets/ref_table1.png)

### Reproduction Results

Here are some examples of cropped and black image:

![alt text](assets/example.png)

*the following image summarizes the reproduction results:*

![Reproduction Results](assets/result.png)


