# RSFMs (Remote Sensing Foundation Models) Playground

Foundation models have quickly become the de facto standard for developing remote sensing applications. New models come out every week and the [Awesome-Remote-Sensing-Foundation-Models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models) is a great reference that keeps track of them (shout-out to @Jack-bo1220 and @danielz02). 
 
Nevertheless, at Synativ, we missed a framework for quickly evaluating the various models on different public and proprietary datasets. Every time we set out to experiment with the latest model, we spent a week to make the model repo run efficiently within our environment. 

We have heard that fellow scientist and engineers struggle with the same issues and that is why we decided to start building this open-source, independent playground for RSFMs. It offers a robust framework for fine-tuning RSFMs on various use-case specific datasets.

We welcome contributors to expand this playground together with us and make it as useful as possible to the community.

## Contributing

Feel free to contribute to this repo by opening PRs, issues, or discussion topics. We would love to hear which foundation models or datasets you would like to see added for your own experimentation. 

If you would like to get in touch with us either send us an email [info@synativ.com](mailto:info@synativ.com) or reach out on [X](https://x.com/synativ).


## Roadmap

We aim to expand this repository with valuable datasets & foundation models. There is quite a backlog and we would like to set a roadmap together with the community. Some requests that we have received so far:

- [ ] Config files for quickly fine-tuning and testing Clay's foundation model.
- [ ] Including SAR foundation models, specifically SARATR-X.
- [ ] Dataset loaders for Satellogic's dataset.

Please let us know what would be valuable to you and we will put it on the roadmap.


## Installation

Assuming you have an environment with `pip>=21.8` and `python>=3.10`, there are three steps to go through to install `rsfms`:

1. `conda install -c conda-forge gdal`
2. `pip install -r requirements.txt`
3. `pip install -e .`


## Quick start

It is advised to start fine-tuning / testing by using the LightningCLI and providing it the required config files.

To fine-tune use the following command:

```
rsfms fit --config configs/experiment.yaml \
          --data configs/<pick_your_dataset>.yaml \
          --model configs/<pick_your_foundation_model>.yaml
```

To test use the following command:

```
rsfms test --ckpt_path <path_to_your_checkpoint>.pt \
           --config ../configs/experiment.yaml \ 
           --data ../configs/datasets/<pick_your_dataset>.yaml \
           --model ../configs/foundation_models/<pick_your_foundation_model>.yaml
```

We have provided scripts in the `scripts` folder to run these commands.


## Acknowledgement

This project includes software developed by TerraTorch and others. Credits go to the original authors.
