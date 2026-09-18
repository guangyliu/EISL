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

## Pretrained models and processed data
The BART checkpoints and the pre-noised Multi30k archives that used to be downloadable from
a university SharePoint are no longer available. The noisy training targets are regenerated
from Multi30k with `bk_shell/generate_noisy_MT.py` / `bk_shell/generate_noise.sh` (shuffle,
repetition, blank, synthetic noise; see [examples/translation](examples/translation) for the
base preprocessing), and the
models retrain with the run scripts in this folder.

## Generated Results
The generated files from test set are in [log/hypo/hypo](log/hypo/hypo). The original source is *test.de* and the target is *test.en*. For different noise (e.g., shuffle), \*hypo.txt is the generated files of different loss and different scale of noise, and \*bleu is the BLEU score of each target sentence.

