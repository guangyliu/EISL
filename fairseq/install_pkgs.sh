pip install --editable ./
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..
pip install -U git+https://github.com/ddkang/loss_dropper.git
pip install sacremoses tensorboard
pip install sacrebleu==1.5.1 