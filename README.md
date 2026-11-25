# SKME-DETR for cigarette detection

SKME-DETR locates cigarettes in images and returns a bounding box and confidence score for each detection. This repository accompanies the paper *SKME-DETR: An Enhanced End-to-End Transformer for Cigarette Detection in Smoking Monitoring*. It contains the model definition, the WSMK dataset configuration, and the code to train and evaluate the detector.

## Model

For a detailed description of the model architecture, refer to SKME-DETR: An Enhanced End-to-End Transformer for Cigarette Detection in Smoking Monitoring. 

The network is defined in [SKME_DETR.yaml](SKME_DETR.yaml).


## Installation

Run the commands below from the directory containing [train.py](train.py).  For GPU use, install a CUDA-enabled PyTorch build suitable for your machine first. Install the project dependencies and `dill`, which the supplied checkpoint needs when loading.

```shell
python -m pip install -r requirements.txt dill
```

## WSMK dataset

[dataset/data.yaml](dataset/data.yaml) defines one class, `0 = cigarette`. 

| WSMK split | Images in this project | Cigarette boxes reported in the paper (Table I) |
| --- | ---: | ---: |
| Train | 5,247 | 5,609 |
| Validation | 749 | 797 |
| Test | 1,500 | 1,591 |
| **Total** | **7,496** | **7,997** |



```text
dataset/
  data.yaml
  WSMK/
    images/{train,val,test}/
    labels/{train,val,test}/
```



## Training

```shell
python train.py
```

The script trains SKME-DETR from scratch, takes the class count from the dataset, and trains with AdamW for 300 epochs at 640 × 640. Its defaults are a batch size of 16 and one data-loading worker. 



## Evaluation

From the project root, [test.py](test.py) loads `best.pt` and evaluates the **test** split at 640 × 640 with a batch size of 32. 

```shell
python test.py
```

Test outputs include `predictions.json`, precision–recall curves and labeled prediction examples under `runs/test/`.



## License

The project uses the [GNU Affero General Public License v3.0](LICENSE).
