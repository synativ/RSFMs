#!/bin/bash

rsfms test --ckpt_path <path_to_your_checkpoint>.pt \
           --config ../configs/experiment.yaml \ 
           --data ../configs/datasets/<pick_your_dataset>.yaml \
           --model ../configs/foundation_models/<pick_your_foundation_model>.yaml

