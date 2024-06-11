#!/bin/bash

rsfms fit --config ../configs/experiment.yaml \
          --data ../configs/datasets/<pick_your_dataset>.yaml \
          --model ../configs/foundation_models/<pick_your_foundation_model>.yaml
