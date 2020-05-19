---   
<div align="center">    
 
# Plan2Explore    

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2005.05960-red.svg)](https://arxiv.org/pdf/2005.05960.pdf)
[![Status](https://img.shields.io/badge/Status-Incomplete-red.svg)]()

</div>
 
## Description   
pytorch implementation of the Plan2Expore algorithm for environments with low dimensional observations

### Differences from the Original

- adapted impementation for low dimensional observations(no RSSM, Encoder, Decoder, Latent Space)
- disagreement ensamble is made up of state transition models DE(s_t, a) -> s_t+1

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/Gregory-Eales/Plan2Explore  

# install project   
cd Plan2Explore 
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to plan2explore, train and then run it.   
 ```bash
# module folder
cd src/    

# train model
python train.py

# run model
python run.py
```


## Results


### Source  
- [Original Author Repo](https://github.com/ramanans1/plan2explore) 

```
@inproceedings{sekar20plan2explore,
    Author = {Sekar, Ramanan and Rybkin, Oleh and Daniilidis, Kostas and
              Abbeel, Pieter and Hafner, Danijar and Pathak, Deepak},
    Title = {Planning to Explore via Self-Supervised World Models},
    Booktitle = {},
    Year = {2020}
}
```   
