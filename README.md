# Vistec-AIS Speech Emotion Recognition
![python-badge](https://img.shields.io/badge/python-%3E%3D3.6-blue?logo=python)
![pytorch-badge](https://img.shields.io/badge/pytorch-%3E%3D1.7.1-red?logo=pytorch)
![license](	https://img.shields.io/github/license/tann9949/vistec-ser)

[comment]: <> (![Upload Python Package]&#40;https://github.com/tann9949/vistec-ser/workflows/Upload%20Python%20Package/badge.svg&#41;)

[comment]: <> (![Training]&#40;https://github.com/tann9949/vistec-ser/workflows/Training/badge.svg&#41;)

![Code Grade](https://www.code-inspector.com/project/17426/status/svg)
![Code Quality Score](https://www.code-inspector.com/project/17426/score/svg)

Speech Emotion Recognition Model and Inferencing using Pytorch

## Installation
### From Pypi
```shell
pip install vistec-ser
```

### From source
```shell
git clone https://github.com/tann9949/vistec-ser.git
cd vistec-ser
python setup.py install
```

## Usage
### Training with AIS-SER-TH Dataset
We provide Google Colaboratory example for training the [AIS-SER-TH dataset]() using our repository.

[![colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1kF5xBYe7d48JRaz3KfIK65A4N5dZMqWQ?usp=sharing)

### Training using provided scripts
Note that currently, this workflow only supports pre-loaded features. So it might comsume an additional overhead of ~2 Gb or RAM. To 
run the experiment. Run the following command

Since there are 80 studios recording and 20 zoom recording. We split the dataset into 10-fold, 10 studios each. Then evaluate using
k-fold cross validation method. We provide 2 k-fold experiments: including and excluding zoom recording. This can be configured 
in config file (see `examples/aisser.yaml`)

```shell
python examples/train_fold_aisser.py --config-path <path-to-config> --n-iter <number-of-iterations>  
```

### Inferencing
We also implement a FastAPI backend server as an example of deploying a SER model. To run the server, run
```shell
cd examples
uvicorn server:app --reload
```
You can customize the server by modifying `example/thaiser.yaml` in `inference` field.

Once the server spawn, you can do HTTP POST request in `form-data` format. and JSON will return as the following format:
```json
[
  {
    "name": <request-file-name>,
    "prob": {
      "neutral": <p(neu)>,
      "anger": <p(ang)>,
      "happiness": <p(hap)>,
      "sadness": <p(sad)>
    }
  }, ...
]
```
See an example below:

![server-demo](figures/server.gif)

## Author & Sponsor
<a href="https://airesearch.in.th/" style="margin-right:50px">
<img src="https://airesearch.in.th/assets/img/logo/airesearch-logo.svg" alt="airesearch" width="200"/>
</a>
<a href="https://www.ais.co.th/">
<img src="https://upload.wikimedia.org/wikipedia/en/thumb/3/3b/Advanced_Info_Service_logo.svg/1200px-Advanced_Info_Service_logo.svg.png" alt="ais" width="200"/>
</a>

Chompakorn Chaksangchaichot

Email: [chompakornc_pro@vistec.ac.th](`chompakornc_pro@vistec.ac.th)