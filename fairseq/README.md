Attention: below commands should be executed under fairseq folder.
## Enviroments
We recommend to create a new conda enviroment (named eisl):
```shell
conda create -n eisl python==3.7  pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch

```
Then activate the conda enviroment:
```shell
conda activate eisl
```
Install the required package by running the script:
```shell
bash install_pkgs.sh
```

## Download the Pretrained Models
For Multi30k dataset, we provide all the trained models (BART) mentioned in the Section 4.1. You can run the below command to download and extract the models ($noise should be one of *shuffle, repetition, blank, multiple*)
```shell
cd ckpts 
bash download_models.sh $noise
```

## Download the Processed Datasets 
You can download the processed data (noisy Multi30k data) by run the script
```shell
bash download_datasets.sh
```

## Code
Work In Progress

## Generated Results
The generated files from test set are in [log/hypo/hypo](log/hypo/hypo). The original source is *test.de* and the target is *test.en*. For different noise (e.g., shuffle), \*hypo.txt is the generated files of different loss and different scale of noise, and \*bleu is the BLEU score of each target sentence.

