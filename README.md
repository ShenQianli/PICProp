### PICProp: Physics-Informed Confidence Propagation for Uncertainty Quantification

This is the official implementation of PICProp: Physics-Informed Confidence Propagation for Uncertainty Quantification 
(NeurIPS 2023).

Arxiv: https://arxiv.org/abs/2310.06923

Please feel free to shoot an email to [shenqianli@u.nus.edu](shenqianli@u.nus.edu) for any questions. 

#### Usage:

Install required python packages:

```bash
pip install -r requirements.txt
```
Add the `your/path/to/PICProp` to `$PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:your/path/to/PICProp
```

Enter desired experiment directory, for example:

```bash
cd your/path/to/PICProp/experiment/pedagogical
```

Execute scripts to start training. For PICProp in a brute-force manner:

```bash
python run_picprop.py
```

For EffiPICProp, 

```bash
python run_effipicprop.py --lamb $lamb
```

The script will automatically sweep required query points and start a PICProp run if the result is absent, and summarize
the result in a basic plot.

