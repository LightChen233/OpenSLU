# Configuation

## 1. Introduction

Configuration is divided into fine-grained reusable modules:

- `base`: basic configuration
- `logger`: logger setting
- `model_manager`: loading and saving model parameters
- `accelerator`: whether to enable multi-GPU
- `dataset`: dataset management
- `evaluator`: evaluation and metrics setting.
- `tokenizer`: Tokenizer initiation and tokenizing setting.
- `optimizer`: Optimizer initiation setting.
- `scheduler`: scheduler initiation setting.
- `model`: model construction setting.

From Sec. 2 to Sec. 11, we will describe the configuration in detail. Or you can see [Examples](examples/README.md) for Quick Start.

NOTE: `_*_` config are reserved fields in OpenSLU.

## Configuration Item Script
In OpenSLU configuration, we support simple calculation script for each configuration item. For example, we can get `dataset_name` by using `{dataset.dataset_name}`, and fill its value into python script `'LightChen2333/agif-slu-' + '*'`.(Without '', `{dataset.dataset_name}` value will be treated as a variable).

NOTE: each item with `{}` will be treated as python script. 
```yaml
tokenizer:
  _from_pretrained_: "'LightChen2333/agif-slu-' + '{dataset.dataset_name}'"  # Support simple calculation script

```

## `base` Config
```yaml
# `start_time` will generated automatically when start any config script, needless to be assigned.
# start_time: xxxxxxxx               
base:
  name: "OpenSLU"                  # project/logger name
  multi_intent: false              # whether to enable multi-intent setting
  train: True                      # enable train else enable zero-shot
  test: True                       # enable test during train.
  device: cuda                     # device for cuda/cpu
  seed: 42                         # random seed
  best_key: EMA                    # save model by which metric[intent_acc/slot_f1/EMA]
  tokenizer_name: word_tokenizer   # tokenizer: word_tokenizer for no pretrained model, else use [AutoTokenizer] tokenizer name
  add_special_tokens: false        # whether add [CLS], [SEP] special tokens
  epoch_num: 300                   # train epoch num
#  eval_step: 280                  # if eval_by_epoch = false and eval_step > 0, will evaluate model by steps
  eval_by_epoch: true              # evaluate model by epoch
  batch_size: 16                   # batch size
```
## `logger` Config
```yaml
logger:
  # `wandb` is supported both in single- multi-GPU,
  # `tensorboard` is only supported in multi-GPU,
  # and `fitlog` is only supported in single-GPU
  logger_type: wandb 
```
## `model_manager` Config
```yaml
model_manager:
  # if load_dir != `null`, OpenSLU will try to load checkpoint to continue training,
  # if load_dir == `null`, OpenSLU will restart training.
  load_dir: null
  # The dir path to save model and training state.
  # if save_dir == `null` model will be saved to `save/{start_time}`
  save_dir: save/stack
  # save_mode can be selected in [save-by-step, save-by-eval]
  # `save-by-step` means save model only by {save_step} steps without evaluation.
  # `save-by-eval` means save model by best validation performance
  save_mode: save-by-eval 
  # save_step: 100         # only enabled when save_mode == `save-by-step`
  max_save_num: 1          # The number of best models will be saved.
```
## `accelerator` Config
```yaml
accelerator:
  use_accelerator: false   # will enable `accelerator` if use_accelerator is `true`
```
## `dataset` Config
```yaml
dataset:
  # support load model from hugging-face.
  # dataset_name can be selected in [atis, snips, mix-atis, mix-snips]
  dataset_name: atis
  # support assign any one of dataset path and other dataset split is the same as split in `dataset_name`
  # train: atis # support load model from hugging-face or assigned local data path.
  # validation: {root}/ATIS/dev.jsonl 
  # test: {root}/ATIS/test.jsonl
```
## `evaluator` Config
```yaml
evaluator:
  best_key: EMA        # the metric to judge the best model
  eval_by_epoch: true   # Evaluate after an epoch if `true`.
  # Evaluate after {eval_step} steps if eval_by_epoch == `false`.
  # eval_step: 1800
  # metric is supported the metric as below:
  # - intent_acc
  # - slot_f1
  # - EMA
  # - intent_f1
  # - macro_intent_f1
  # - micro_intent_f1
  # NOTE: [intent_f1, macro_intent_f1, micro_intent_f1] is only supported in multi-intent setting. intent_f1 and macro_intent_f1 is the same metric.
  metric:
    - intent_acc
    - slot_f1
    - EMA
```
## `tokenizer` Config
```yaml
tokenizer:
  # Init tokenizer. Support `word_tokenizer` and other tokenizers in huggingface.
    _tokenizer_name_: word_tokenizer 
    # if `_tokenizer_name_` is not assigned, you can load pretrained tokenizer from hugging-face.
    # _from_pretrained_: LightChen2333/stack-propagation-slu-atis
    _padding_side_: right            # the padding side of tokenizer, support [left/ right]
    # Align mode between text and slot, support [fast/ general],
    # `general` is supported in most tokenizer, `fast` is supported only in small portion of tokenizers.
    _align_mode_: fast
    _to_lower_case_: true
    add_special_tokens: false        # other tokenizer args, you can add other args to tokenizer initialization except `_*_` format args
    max_length: 512

```
## `optimizer` Config
```yaml
optimizer:
  _model_target_: torch.optim.Adam # Optimizer class/ function return Optimizer object
  _model_partial_: true            # partial load configuration. Here will add model.parameters() to complete all Optimizer parameters
  lr: 0.001                        # learning rate
  weight_decay: 1e-6               # weight decay
```
## `scheduler` Config
```yaml
scheduler:
  _model_target_: transformers.get_scheduler
  _model_partial_: true     # partial load configuration. Here will add optimizer, num_training_steps to complete all Optimizer parameters
  name : "linear"
  num_warmup_steps: 0
```
## `model` Config
```yaml
model:
  # _from_pretrained_: LightChen2333/stack-propagation-slu-atis # load model from hugging-face and is not need to assigned any parameters below.
  _model_target_: model.OpenSLUModel # the general model class, can automatically build the model through configuration.

  encoder:
    _model_target_: model.encoder.AutoEncoder # auto-encoder to autoload provided encoder model
    encoder_name: self-attention-lstm         # support [lstm/ self-attention-lstm] and other pretrained models those hugging-face supported

    embedding:                                # word embedding layer
#      load_embedding_name: glove.6B.300d.txt  # support autoload glove embedding.  
      embedding_dim: 256                      # embedding dim
      dropout_rate: 0.5                       # dropout ratio after embedding

    lstm:
      layer_num: 1                           # lstm configuration
      bidirectional: true
      output_dim: 256                        # module should set output_dim for autoload input_dim in next module. You can also set input_dim manually.
      dropout_rate: 0.5

    attention:                              # self-attention configuration
      hidden_dim: 1024
      output_dim: 128
      dropout_rate: 0.5

    return_with_input: true                # add inputs information, like attention_mask, to decoder module.
    return_sentence_level_hidden: false    # if return sentence representation to decoder module

  decoder:
    _model_target_: model.decoder.StackPropagationDecoder  # decoder name
    interaction:
      _model_target_: model.decoder.interaction.StackInteraction # interaction module name
      differentiable: false                                      # interaction module config

    intent_classifier:
      _model_target_: model.decoder.classifier.AutoregressiveLSTMClassifier # intent classifier module name
      layer_num: 1
      bidirectional: false
      hidden_dim: 64
      force_ratio: 0.9                                        # teacher-force ratio
      embedding_dim: 8                                        # intent embedding dim
      ignore_index: -100                                      # ignore index to compute loss and metric
      dropout_rate: 0.5
      mode: "token-level-intent"                              # decode mode, support [token-level-intent, intent, slot]
      use_multi: "{base.multi_intent}"
      return_sentence_level: true                             # whether to return sentence level prediction as decoded input

    slot_classifier:
      _model_target_: model.decoder.classifier.AutoregressiveLSTMClassifier
      layer_num: 1
      bidirectional: false
      force_ratio: 0.9
      hidden_dim: 64
      embedding_dim: 32
      ignore_index: -100
      dropout_rate: 0.5
      mode: "slot"
      use_multi: false
      return_sentence_level: false
```

## Implementing a New Model

### 1. Interaction Re-Implement
Here we take `DCA-Net` as an example:

In most cases, you just need to rewrite `Interaction` module:

```python
from common.utils import HiddenData
from model.decoder.interaction import BaseInteraction
class DCANetInteraction(BaseInteraction):
    def __init__(self, **config):
        super().__init__(**config)
        self.T_block1 = I_S_Block(self.config["output_dim"], self.config["attention_dropout"], self.config["num_attention_heads"])
        ...

    def forward(self, encode_hidden: HiddenData, **kwargs):
        ...
```

and then you should configure your module:
```yaml
base:
  ...

optimizer:
  ...

scheduler:
  ...

model:
  _model_target_: model.OpenSLUModel
  encoder:
    _model_target_: model.encoder.AutoEncoder
    encoder_name: lstm

    embedding:
      load_embedding_name: glove.6B.300d.txt
      embedding_dim: 300
      dropout_rate: 0.5

    lstm:
      dropout_rate: 0.5
      output_dim: 128
      layer_num: 2
      bidirectional: true
    output_dim: "{model.encoder.lstm.output_dim}"
    return_with_input: true
    return_sentence_level_hidden: false

  decoder:
    _model_target_: model.decoder.DCANetDecoder
    interaction:
      _model_target_: model.decoder.interaction.DCANetInteraction
      output_dim: "{model.encoder.output_dim}"
      attention_dropout: 0.5
      num_attention_heads: 8

    intent_classifier:
      _model_target_: model.decoder.classifier.LinearClassifier
      mode: "intent"
      input_dim: "{model.decoder.output_dim.output_dim}"
      ignore_index: -100

    slot_classifier:
      _model_target_: model.decoder.classifier.LinearClassifier
      mode: "slot"
      input_dim: "{model.decoder.output_dim.output_dim}"
      ignore_index: -100
```

Oops, you finish all model construction. You can run script as follows to train model:
```shell
python run.py -cp config/dca_net.yaml [-ds atis]
```
### 2. Decoder Re-Implement
Sometimes, `interaction then classification` order can not meet your needs. Therefore, you should simply rewrite decoder for flexible interaction order:

Here, we take `stack-propagation` as an example:
1. We should rewrite interaction module for `stack-propagation`
```python
from common.utils import ClassifierOutputData, HiddenData
from model.decoder.interaction.base_interaction import BaseInteraction
class StackInteraction(BaseInteraction):
    def __init__(self, **config):
        super().__init__(**config)
        ...

    def forward(self, intent_output: ClassifierOutputData, encode_hidden: HiddenData):
        ...
```
2. We should rewrite `StackPropagationDecoder` for stack-propagation interaction order:
```python
from common.utils import HiddenData, OutputData
class StackPropagationDecoder(BaseDecoder):

    def forward(self, hidden: HiddenData):
        pred_intent = self.intent_classifier(hidden)
        hidden = self.interaction(pred_intent, hidden)
        pred_slot = self.slot_classifier(hidden)
        return OutputData(pred_intent, pred_slot)
```

3. Then we can easily combine general model by `config/stack-propagation.yaml` configuration file:
```yaml
base:
  ...

...

model:
  _model_target_: model.OpenSLUModel

  encoder:
    ...

  decoder:
    _model_target_: model.decoder.StackPropagationDecoder
    interaction:
      _model_target_: model.decoder.interaction.StackInteraction
      differentiable: false

    intent_classifier:
      _model_target_: model.decoder.classifier.AutoregressiveLSTMClassifier
      ... # parameters needed __init__(*)
      mode: "token-level-intent"
      use_multi: false
      return_sentence_level: true

    slot_classifier:
      _model_target_: model.decoder.classifier.AutoregressiveLSTMClassifier
      ... # parameters needed __init__(*)
      mode: "slot"
      use_multi: false
      return_sentence_level: false
```
4. You can run script as follows to train model:
```shell
python run.py -cp config/stack-propagation.yaml
```



