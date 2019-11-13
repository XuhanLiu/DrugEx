#!/usr/bin/env bash

# The goal of this bash script is to illustrate
# how to run the aforementioned scripts to obtain an optimal
# generative model based on the information from
# the DrugEx paper (https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0355-6).
#
# That is:
#
# - RF classifier as reward in RL
# - 300 epochs for the pre-trained (exploitation) network
#   and 400 for the fine-tuned (exploration) network (Fig. 5)
# - 200 epochs during the RL training (Fig. 8) with
#   the fine-tuned network as exploration strategy (Gφ),
#   ε = 0.01 and β = 0.1 (based on Table 1)

# data assembly and training
python dataset.py
python environ.py
python pretrainer.py
python agent.py -e 0.01 -b 0.1

# use the trained model to sample 1000 molecules
python designer.py -i 'output/net_e_0.01_0.1_500x10.pkg' -n 1000