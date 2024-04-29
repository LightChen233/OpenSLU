<img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/OpenSLU.jpg" alt=""/>

---

<p align="center">
  	<a >
      <img alt="version" src="https://img.shields.io/badge/version-v0.1.0-blue?color=FF8000?color=009922" />
    </a>
  <a href=""><img src="https://img.shields.io/badge/python-3.6.2+-orange.svg"></a>
  <a >
       <img alt="PRs-Welcome" src="https://img.shields.io/badge/PRs-Welcome-red" />
  	</a>
   	<a>
       <img alt="stars" src="https://img.shields.io/github/stars/LightChen233/OpenSLU" />
  	</a>
  	<a href="https://github.com/LightChen233/OpenSLU/network/members">
       <img alt="FORK" src="https://img.shields.io/github/forks/LightChen233/OpenSLU?color=FF8000" />
  	</a>
    <a href="https://github.com/LightChen233/OpenSLU/issues">
      <img alt="Issues" src="https://img.shields.io/github/issues/LightChen233/OpenSLU?color=0088ff"/>
    </a>
    <br />
</p>

<div>
<img src="./img/csu_logo.png" width="48%">
<img src="./img/SCIR_logo.png" width="48%">
</div>

## 🔥News
- \[04/30\] 🔥 We support a series of new SLU algorithms, including Co-guiding, MISCA, GIS-Co, DPF.

## <img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/motivation.png" width="25" />  Motivation

Spoken Language Understanding (SLU) is one of the core components of a task-oriented dialogue system, which aims to extract the semantic meaning of user queries (e.g., intents and slots).

In this work, we introduce __OpenSLU__, an open-source toolkit to provide a uniﬁed, modularized, and extensible toolkit for spoken language understanding. Speciﬁcally, OpenSLU uniﬁes 10 SLU baselines for both single-intent and multi-intent scenarios, which support both non-pretrained and pretrained models simultaneously. Additionally, OpenSLU is highly modularized and extensible by decomposing the model architecture, inference, and learning process into reusable modules, which allows researchers to quickly set up SLU experiments with highly ﬂexible conﬁgurations. We hope OpenSLU can help researcher to quickly initiate experiments and spur more breakthroughs in SLU.

## <img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/notes.png" width="25" /> Reference

If you find this project useful for your research, please consider citing the following paper:

```
@inproceedings{qin-etal-2023-openslu,
    title = "{O}pen{SLU}: A Unified, Modularized, and Extensible Toolkit for Spoken Language Understanding",
    author = "Qin, Libo  and
      Chen, Qiguang  and
      Xu, Xiao  and
      Feng, Yunlong  and
      Che, Wanxiang",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-demo.9",
    doi = "10.18653/v1/2023.acl-demo.9",
    pages = "95--102",
}
```



## <img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/resource.png" width="25" /> Installation
### System requirements
OpenSLU requires `Python>=3.8`, and `torch>=1.12.0`.
### Install from git
```bash 
git clone https://github.com/LightChen233/OpenSLU.git && cd OpenSLU/
pip install -r requirements.txt
```


## File Structure

```yaml
root
├── common
│   ├── config.py           # load configuration and auto preprocess ignored config
│   ├── global_pool.py           # global variable pool, you can use set_value() to add variable into pool and get_value() to get variable from pool.
│   ├── loader.py           # load data from hugging face
│   ├── logger.py           # log predict result, support [fitlog], [wandb], [local logging]
│   ├── metric.py           # evalutation metric, support [intent acc], [slot F1], [EMA]
│   ├── model_manager.py    # help to prepare data, prebuild training progress.
│   ├── saver.py    # help to manage to save model, checkpoint etc. to disk.
│   ├── tokenizer.py        # tokenizer also support no-pretrained model for word tokenizer.
│   └── utils.py            # canonical model communication data structure and other common tool function
├── config
│   ├── reproduction        # configurations for reproducted SLU model. 
│   └── **.yaml             # configuration for SLU model.
├── logs                    # local log storage dir path.
├── model
│   ├── encoder
│   │   ├── base_encoder.py          # base encoder model. All implemented encoder models need to inherit the BaseEncoder class
│   │   ├── auto_encoder.py          # auto-encoder to autoload provided encoder model
│   │   ├── non_pretrained_encoder.py # all common-used no pretrained encoder like lstm, lstm+self-attention
│   │   └── pretrained_encoder.py    # all common-used pretrained encoder, implemented by hugging-face [AutoModel].
│   ├── decoder
│   │   ├── interaction
│   │   │   ├── base_interaction.py  # base interaction model. All implemented encoder models need to inherit the BaseInteraction class
│   │   │   └── *_interaction.py     # some SOTA SLU interaction module. You can easily reuse or rewrite to implement your own idea.
│   │   ├── base_decoder.py # decoder class, [BaseDecoder] support classification after interaction, also you can rewrite for your own interaction order 
│   │   └── classifier.py   # classifier class, support linear and LSTM classification. Also support token-level intent.
│   └── open_slu_model.py   # the general model class, can automatically build the model through configuration.
├── save                    # model checkpoint storage dir path and dir to automatically save glove embedding.
├── tools                         # some callable tools
│   ├── load_from_hugging_face.py # apis to help load checkpoint to reproduction from hugging face.
│   ├── parse_to_hugging_face.py  # help to convert checkpoint to hugging face needed format.
│   └── visualization.py          # help to visualize prediction error.
├── app.py                    # help to deploy model in hugging face space. 
└── run.py                  # run script for all function.
```

## <img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/catalogue.png" width="27" /> Quick Start

### 1. Reproducing Existing Models
Example for reproduction of `slot-gated` model:

```bash
python run.py --dataset atis --model slot-gated 
```
**NOTE**: The configuration files of these models are placed in
`config/reproducion/*`.

### 2. Customizable Combination Existing Components
1. First, you can freely combine and build your own model through config files. For details, see [Configuration](config/README.md).
2. Then, you can assign the configuration path to train your own model. 

Example for `stack-propagation` fine-tuning:

```bash
python run.py -cp config/stack-propagation.yaml
```

Example for multi-GPU fine-tuning:

```bash
accelerate config
accelerate launch run.py -cp config/stack-propagation.yaml
```

Or you can assign `accelerate` yaml configuration.

```bash
accelerate launch [--config_file ./accelerate/config.yaml] run.py -cp config/stack-propagation.yaml
```

### 3. Implementing a New SLU Model
In OpenSLU, you are only needed to rewrite required commponents and assign them in configuration instead of rewriting all commponents.

In most cases, rewriting Interaction module is enough for building a new SLU model.
This module accepts [HiddenData](./common/utils.py) as input and return with `HiddenData`, which contains the `hidden_states` for `intent` and `slot`, and other helpful information. The example is as follows:
```python
class NewInteraction(BaseInteraction):
    def __init__(self, **config):
        self.config = config
        ...
    
    def forward(self, hiddens: HiddenData):
        ...
        intent, slot = self.func(hiddens)
        hiddens.update_slot_hidden_state(slot)
        hiddens.update_intent_hidden_state(intent)
        return hiddens
```

To further meet the
needs of complex exploration, we provide the
[BaseDecoder](./model/decoder/base_decoder.py) class, and the user can simply override the `forward()` function in class, which accepts `HiddenData` as input and `OutputData` as output. The example is as follows:
```python
class NewDecoder(BaseDecoder):
    def __init__(self,
                intent_classifier,
                slot_classifier,
              interaction=None):
        ...
        self.int_cls = intent_classifier
        self.slot_cls = slot_classifier
        self.interaction = interaction
        
    def forward(self, hiddens: HiddenData):
        ...
        interact = self.interaction(hiddens)
        slot = self.slot_cls(interact.slot)
        intent = self.int_cls(interact.intent)
        return OutputData(intent, slot)
```

**NOTE**: We have set "logger" to global_pool, user can use `global_pool.get_value("logger")` to get logger module, and call any interface for logger anywhere.

### 4. Customizing your data

OpenSLU require the input file to be in the `.jsonl` format, with each line representing a `JSON` object. The format should adhere to the following structure: 

```json
{
    "text": ["A", "model", "bilayer", "can", "be", "made", "with", "either", "synthetic", "or", "natural", "lipids", "."], 
    "intent": "XXX", 
    "slot": ["O", "B-X", "I-X", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]
}
```

After that, you will need to specify the corresponding data path at [`dataset`](config/README.md) configuration.

## <img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/folders.png" width="25" /> Modules

### 1. Encoder Modules

- **No Pretrained Encoder**
  - GloVe Embedding
  - BiLSTM Encoder
  - BiLSTM + Self-Attention Encoder
  - Bi-Encoder (support two encoders for intent and slot, respectively)  
- **Pretrained Encoder**
  - `bert-base-uncased`
  - `roberta-base`
  - `microsoft/deberta-v3-base`
  - other hugging-face supported encoder model...

### 2. Decoder Modules

#### 2.1 Interaction Modules

- DCA Net Interaction
- Stack Propagation Interaction
- Bi-Model Interaction(with decoder/without decoder)
- Slot Gated Interaction

#### 2.2 Classification Modules
All classifier support `Token-level Intent` and `Sentence-level intent`. What's more, our decode function supports to both `Single-Intent` and `Multi-Intent`.
- LinearClassifier
- AutoregressiveLSTMClassifier
- MLPClassifier

### 3. Supported Models
We implement various 10 common-used SLU baselines:

---
**Single-Intent Model**
- Bi-Model  \[ [Wang et al., 2018](https://aclanthology.org/N18-2050/) \] : 
  - `bi-model.yaml`
- Slot-Gated \[ [Goo et al., 2018](https://www.csie.ntu.edu.tw/~yvchen/doc/NAACL18_SlotGated.pdf) \] : 
  - `slot-gated.yaml`
- Stack-Propagation \[ [Qin et al., 2019](https://www.aclweb.org/anthology/D19-1214/) \] : 
  - `stack-propagation.yaml`
- Joint Bert \[ [Chen et al., 2019](https://arxiv.org/abs/1902.10909) \] : 
  - `joint-bert.yaml`
- RoBERTa \[ [Liu et al., 2019](https://arxiv.org/abs/1907.11692) \] : 
  - `roberta.yaml`
- ELECTRA \[ [Clark et al., 2020](https://arxiv.org/abs/2003.10555) \] : 
  - `electra.yaml`
- DCA-Net \[ [Qin et al., 2021](https://arxiv.org/abs/2010.03880) \] : 
  - `dca_net.yaml`
- DeBERTa \[ [He et al., 2021](https://arxiv.org/abs/2111.09543) \] : 
  - `deberta.yaml`

---

<table style="text-align:center;">
	<tr>
	    <th rowspan="2">Model</th>
	    <th colspan="3"> ATIS </th>
	    <th colspan="3">SNIPS</th>
	</tr >
  <tr>
	    <th> Slot F1.(%) </th>
	    <th> Intent Acc.(%) </th>
      <th> EMA(%) </th>
      <th> Slot F1.(%) </th>
	    <th> Intent Acc.(%) </th>
      <th> EMA(%) </th>
	</tr >
  <tr>
	    <td colspan="7"> Non-Pretrained Models
	</td >
  <tr >
        <td style="text-align:left;">Slot Gated [<a href="https://www.csie.ntu.edu.tw/~yvchen/doc/NAACL18_SlotGated.pdf">Goo et al., 2018</a>]</td>
	    <td>94.7</td>
        <td>94.5</td>
        <td>82.5</td>
        <td>93.2</td>
        <td>97.6</td>
        <td>85.1</td>
	</tr>
	<tr >
        <td style="text-align:left;">Bi-Model [<a href="https://aclanthology.org/N18-2050/">Wang et al., 2018</a>]</td>
	    <td>95.2</td>
        <td>96.2</td>
        <td>85.6</td>
        <td> 93.1</td>
        <td>97.6 </td>
        <td>84.1 </td>
	</tr>
  <tr >
        <td style="text-align:left;">Stack Propagation [<a href="https://www.aclweb.org/anthology/D19-1214/">Qin et al., 2019</a>]</td>
	    <td>95.4</td>
        <td>96.9</td>
        <td>85.9</td>
        <td><b>94.6</b></td>
        <td>97.9</td>
        <td>87.1</td>
	</tr>
  <tr >
        <td style="text-align:left;">DCA Net [<a href="https://arxiv.org/abs/2010.03880">Qin et al., 2021</a>]</td>
	    <td><b>95.9</b></td>
        <td><b>97.3</b></td>
        <td><b>87.6</b></td>
        <td>94.3</td>
        <td><b>98.1</b></td>
        <td><b>87.3</b></td>
	</tr>
	<tr>
	    <td colspan="7" style="text-align:center;"> Pretrained Models
	</td >
  </tr>
	<tr>
        <td style="text-align:left;">Joint BERT [<a href="https://arxiv.org/abs/1902.10909">Chen et al., 2019</a>]</td>
	    <td><b>95.8</b></td>
      <td><b>97.9</b></td>
      <td><b>88.6</b></td>
      <td>96.4</td>
      <td>98.4</td>
      <td>91.9</td>
	</tr>
	<tr>
        <td style="text-align:left;">RoBERTa [<a href="https://arxiv.org/abs/1907.11692">Liu et al., 2019</a>]</td>
      <td>95.8</td>
      <td>97.8</td>
      <td>88.1</td>
      <td>95.7</td>
      <td>98.1</td>
      <td>90.6</td>
  </tr>
	<tr>
        <td style="text-align:left;">Electra [<a href="https://arxiv.org/abs/2003.10555">Clark et al., 2020</a>]</td>
      <td>95.8</td>
      <td>96.9</td>
      <td>87.1</td>
      <td>95.7</td>
      <td>98.3</td>
      <td>90.1</td>
  </tr>
	<tr>
			<td style="text-align:left;">DeBERTa<SUB>v3</SUB>[<a href="https://arxiv.org/abs/2111.09543">He et al., 2021</a>]</td>
      <td>95.8</td>
      <td>97.8</td>
      <td>88.4</td>
      <td><b>97.0</b></td>
      <td><b>98.4</b></td>
      <td><b>92.7</b></td>
	</tr>
	
</table>

---

**Multi-Intent Model**
- AGIF \[ [Qin et al., 2020](https://arxiv.org/pdf/2004.10087.pdf) \] : 
  - `agif.yaml`
- GL-GIN \[ [Qin et al., 2021](https://arxiv.org/abs/2106.01925) \] : 
  - `gl-gin.yaml`
- Co-Guiding \[ [Xing et al., 2022](https://aclanthology.org/2022.emnlp-main.12) \] : 
  - `deberta_co_guiding.yaml`
- GIS-Co \[ [Song et al., 2022](https://aclanthology.org/2022.emnlp-main.543) \] : 
  - `deberta_gis_co.yaml`
- MISCA \[ [Pham et al., 2023](https://aclanthology.org/2023.findings-emnlp.841) \] : 
  - `deberta_misca.yaml`
- DPF \[ [Qin et al., 2023]() \] : 
  - Please follow pretraining process at [DPF](https://github.com/LightChen233/DPF) 
---

<table style="text-align:center;">
	<tr>
	    <th rowspan="2">Model</th>
	    <th colspan="4"> Mix-ATIS </th>
	    <th colspan="4">Mix-SNIPS</th>
	</tr >
  <tr>
	    <th> Slot F1.(%) </th>
      <th> Intent F1.(%) </th>
	    <th> Intent Acc.(%) </th>
      <th> EMA(%) </th>
      <th> Slot F1.(%) </th>
      <th> Intent F1.(%) </th>
	    <th> Intent Acc.(%) </th>
      <th> EMA(%) </th>
	</tr >
  <tr>
	    <td colspan="9"> Non-Pretrained Models
	</td >
  <tr >
        <td style="text-align:left;">Vanilla Multi Task Framework</td>
      <td>85.7</td>
      <td>80.8</td>
      <td>75.4</td>
      <td>36.4</td>
      <td>92.7</td>
      <td>98.3</td>
      <td>96.0</td>
      <td>70.2</td>
	</tr>
	<tr >
        <td style="text-align:left;">AGIF [<a href="https://arxiv.org/pdf/2004.10087.pdf">Qin et al., 2020</a>]</td>
	    <td><b>86.9</b></td>
      <td>80.0</td>
      <td>72.7</td>
      <td>39.5</td>
      <td><b>94.4</b></td>
      <td>97.4</td>
      <td>93.7</td>
      <td>74.8</td>
	</tr>
  <tr >
        <td style="text-align:left;">GL-GIN [<a href="https://arxiv.org/abs/2106.01925">Qin et al., 2021</a>]</td>
	    <td>86.3</td>
      <td><b>81.1</b></td>
      <td><b>77.1</b></td>
      <td><b>43.6</b></td>
      <td>94.2</td>
      <td><b>98.5</b></td>
      <td><b>96.1</b></td>
      <td><b>74.9</b></td>
	</tr>
  <!-- <tr >
	    <td colspan="9"> Pretrained Models
	</td >
  <tr >
  <tr >
        <td style="text-align:left;">Deberta-v3 [<a href="https://openreview.net/forum?id=sE7-XhLxHA">He et al., 2023</a>]</td>
	    <td>86.3</td>
      <td><b>81.1</b></td>
      <td><b>77.1</b></td>
      <td><b>43.6</b></td>
      <td>94.2</td>
      <td><b>98.5</b></td>
      <td><b>96.1</b></td>
      <td><b>74.9</b></td>
	</tr> -->
</table>

---

\* NOTE: Due to some stochastic factors(e.g., GPU and environment), it maybe need to slightly tune the hyper-parameters using grid search to obtain better results.

## <img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/organizer.png" width="25" /> Application
### 1. Visualization Tools
Model metrics tests alone no longer adequately reflect the model's performance. To help researchers further improve their models, we provide a tool for visual error analysis. 

We provide an analysis interface with three main parts: 
- (a) error distribution analysis; 
- (b) label transfer analysis;
- (c) instance analysis.

<img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/visual_analysis.png" />

```bash
python tools/visualization.py \
       --config_path config/visual.yaml \
       --output_path {ckpt_dir}/outputs.jsonl
```
Visualization configuration can be set as below: 
```yaml
host: 127.0.0.1
port: 7861
is_push_to_public: true               # whether to push to gradio platform(public network)
output_path: save/stack/outputs.jsonl # output prediction file path
page-size: 2                          # the number of instances of each page in instance anlysis. 
```
### 2. Deployment

We provide an script to deploy your model automatically. You are only needed to run the command as below to deploy your own model: 

```bash
python app.py --config_path config/examples/from_pretrained.yaml
```

<img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/app.png" />

**NOTE**: Please set `logger_type` to `local` if you do not want to login with `wandb` for deployment.

### 3. Publish your model to hugging face

We also offer an script to transfer models trained by OpenSLU to hugging face format automatically. And you can upload the model to your `Model` space.

```shell
python tools/parse_to_hugging_face.py -cp config/reproduction/atis/bi-model.yaml -op save/temp
```

It will generate 5 files, and you should only need to upload `config.json`, `pytorch_model.bin` and `tokenizer.pkl`.
After that, others can reproduction your model just by adjust `_from_pretrained_` parameters in Configuration.

## <img src="https://cdn.jsdelivr.net/gh/LightChen233/blog-img/intro.png" width="25" /> Contact

Please create Github issues here or email [Libo Qin](mailto:lbqin@ir.hit.edu.cn) or [Qiguang Chen](mailto:charleschen2333@gmail.com) if you have any questions or suggestions. 
