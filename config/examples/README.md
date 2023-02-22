# Examples

Here we introduce some usage of our famework by configuration.

## Reload to train

Firstly, you can run this script to train a `joint-bert` model:
```shell
python run.py -cp config/examples/normal.yaml
```

and you can use `kill` or `Ctrl+C` to kill the training process.

Then, to reload model and continue training, you can run `reload_to_train.yaml` to reload checkpoint and training state.
```shell
python run.py -cp config/examples/reload_to_train.yaml
```

The main difference in `reload_to_train.yaml` is the `model_manager` configuration item:
```yaml
...
model_manager:
  load_train_state: True    # set to True
  load_dir: save/joint_bert # not null
  ...
...
```

## Load from Pre-finetuned model.
We upload all models to [LightChen2333](https://huggingface.co/LightChen2333). You can load those model by simple configuration.
In `from_pretrained.yaml` and `from_pretrained_multi.yaml`, we show two example scripts to load from hugging face in single- and multi-intent, respectively. The key configuration items are as below:
```yaml
tokenizer:
  _from_pretrained_: "'LightChen2333/agif-slu-' + '{dataset.dataset_name}'"  # Support simple calculation script

model:
  _from_pretrained_: "'LightChen2333/agif-slu-' + '{dataset.dataset_name}'" 
```
