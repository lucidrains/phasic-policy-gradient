<img src="./lunar.gif" width="400px"></img>

*1k steps*

## Phasic Policy Gradient - Pytorch

An implementation of Phasic Policy Gradient, a proposed improvement on top of Proximal Policy Optimization (PPO), in Pytorch. It will be my very first project in Reinforcement Learning.

## Install

```bash
$ pip install -r requirements.txt
```

You may need to install `swig`

```bash
$ apt install swig
```

## Use

```bash
$ python train.py --render
```

## Citations

```bibtex
@misc{cobbe2020phasic,
    title   = {Phasic Policy Gradient},
    author  = {Karl Cobbe and Jacob Hilton and Oleg Klimov and John Schulman},
    year    = {2020},
    eprint  = {2009.04416},
    archivePrefix = {arXiv},
    primaryClass = {cs.LG}
}
```

```bibtex
@article{Zhang2024ReLU2WD,
    title   = {ReLU2 Wins: Discovering Efficient Activation Functions for Sparse LLMs},
    author  = {Zhengyan Zhang and Yixin Song and Guanghui Yu and Xu Han and Yankai Lin and Chaojun Xiao and Chenyang Song and Zhiyuan Liu and Zeyu Mi and Maosong Sun},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.03804},
    url     = {https://api.semanticscholar.org/CorpusID:267499856}
}
```

```bibtex
@inproceedings{Lee2024SimBaSB,
    title  = {SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning},
    author = {Hojoon Lee and Dongyoon Hwang and Donghu Kim and Hyunseung Kim and Jun Jet Tai and Kaushik Subramanian and Peter R. Wurman and Jaegul Choo and Peter Stone and Takuma Seno},
    year   = {2024},
    url    = {https://api.semanticscholar.org/CorpusID:273346233}
}
```
