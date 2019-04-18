# Embedding Uncertain Knowledge Graphs

This repository includes the code of UKGE and data used in the experiments.

## Install
Make sure your local environment has the following installed:

    Python3
    tensorflow >= 1.5.0
    scikit-learn
    
Install the dependents using:

    pip install -r requirements.txt

## Run the experiments
To run the experiments, use:

    python ./run/run.py

or

    python ./run/run.py --data ppi5k --model rect --batch_size 1024 --dim 128 --epoch 100 --reg_scale 5e-4
You can use `--model logi` to switch to the UKGE(logi) model.

Data is available at: https://drive.google.com/file/d/1UJQ8hnqPGv1O9pYglfNF5lY_sgDQkleS/view?usp=sharing

## Reference
Please refer to our paper. 
Xuelu Chen, Muhao Chen, Weijia Shi, Yizhou Sun, Carlo Zaniolo. Embedding Uncertain Knowledge Graphs. In *Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence (AAAI)*, 2019

    @inproceedings{chen2019ucgraph,
        title={Embedding Uncertain Knowledge Graphs},
        author={Xuelu Chen, Muhao Chen, Weijia Shi, Yizhou Sun, Carlo Zaniolo},
        booktitle={Proceedings of the Thirty-Third AAAI Conference on Artificial Intelligence (AAAI)},
        year={2019}
    }
