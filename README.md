## PathPredictionForTextClassification

Source code for WWW 2019 paper "Efficient Path Prediction for Semi-Supervised and Weakly Supervised Hierarchical Text Classification"

Readers are welcomed to fork this repository to reproduce the experiments and follow our work. Please kindly cite our paper.

    @article{xiao2019efficient,
    title={Efficient Path Prediction for Semi-Supervised and Weakly Supervised Hierarchical Text Classification},
    author={Xiao, Huiru and Liu, Xin and Song, Yangqiu},
    journal={arXiv preprint arXiv:1902.09347},
    year={2019}
    }



### Requirements
* numpy
* networkx
* scipy
* sklearn
* nltk

### Preprocessing

Filter data with multilabels.
```bash
python filter_multilabels.py
```

Split data to train set and test set.
```bash
python split_data.py
```

Thirdly, generate data managers.
```bash
python build_data_managers.py
```

If you need to generate files in svmlight/libsvm file format, you can run `generate_svmlight_format.py`.
```bash
python generate_svmlight_format.py
```

If you need to run LIBLINEAR, you need to download it and put it in this folder, or create a soft link.

### Training and Prediction

You can run `NB_EM.py`, `LR_SVM.py`, `HierCost.py`, and `LIBLINEAR.py` to train different models.
```bash
python NB_EM.py
python LR_SVM.py
python HierCost.py
python LIBLINEAR.py
```

You can get the results in `data/20ng/0.1` when `settings.label_ratios` include `0.1` and `settings.data_dirs` include `data/20ng`.

**Labeled**

| NB_EM      | flatNB      | levelNB     | NBMC        | TDNB        | WDNB_hard   | PCNB        | flatEM      | levelEM     | EMMC        | TDEM        | WDEM_hard   | PCEM        |
|--------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Macro F1 | 0.726643505 | 0.719444341 | 0.726916676 | 0.709269008 | 0.728670852 | 0.75743725  | 0.75012983  | 0.737743103 | 0.741066577 | 0.715460672 | 0.744308561 | 0.776129457 |
| Micro F1 | 0.789864686 | 0.793181215 | 0.790130008 | 0.789334041 | 0.799416291 | 0.820509419 | 0.821968692 | 0.809896524 | 0.818519501 | 0.799018307 | 0.822234014 | 0.842398514 |

| LR_SVM      | flatLR      | levelLR     | flatSVM     | levelSVM    |
|--------------|-------------|-------------|-------------|-------------|
| Macro F1 | 0.766903969 | 0.765417056 | 0.723047994 | 0.723394201 |
| Micro F1 | 0.803528787 | 0.801671531 | 0.764128416 | 0.765587689 |

| LIBLINEAR      | LIBLINEAR_LR_primal | LIBLINEAR_SVC_primal | LIBLINEAR_SVC_dual |
|--------------|---------------------|-------------------|----------------------|--------------------|
| Macro F1 | 0.773627141         | 0.756098208          | 0.744804057        |
| Micro F1 | 0.811488458           | 0.79145662           | 0.784823561        |


| HierCost      | HierCost_LR | HierCost_ExTrD |
|--------------|-------------|----------------|
| Macro F1 | 0.752839501 | 0.752111229    |
| Micro F1 | 0.791191297 | 0.790925975    |

**Dataless**

| NB_EM      | flatNB      | levelNB     | NBMC        | TDNB        | WDNB_hard   | PCNB        | flatEM      | levelEM     | EMMC        | TDEM        | WDEM_hard   | PCEM        |
|--------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| Macro F1 | 0.455282821 | 0.455198196 | 0.453233098 | 0.416602081 | 0.455843263 | 0.537478122 | 0.474686656 | 0.477696263 | 0.449161501 | 0.427008491 | 0.455843263 | 0.549996369 |
| Micro F1 | 0.600689838 | 0.602812417 | 0.598832582 | 0.58516848  | 0.608384187 | 0.653488989 | 0.626426108 | 0.634253118 | 0.60984346  | 0.612231361 | 0.608384187 | 0.680949854 |

| LR_SVM      | flatLR      | levelLR     | flatSVM     | levelSVM    |
|--------------|-------------|-------------|-------------|-------------|
| Macro F1 | 0.54162369  | 0.52883076  | 0.525801651 | 0.517507715 |
| Micro F1 | 0.644733351 | 0.625232157 | 0.620456354 | 0.607322897 |

| LIBLINEAR  | LIBLINEAR_LR_primal | LIBLINEAR_SVC_primal | LIBLINEAR_SVC_dual |
|----------|---------------------|----------------------|--------------------|
| Macro F1 | 0.536644227         | 0.535923188          | 0.529902503        |
| Micro F1 | 0.641151499         | 0.633987795          | 0.629211993        |

| HierCost      | HierCost_LR | HierCost_ExTrD |
|--------------|-------------|----------------|
| Macro F1 | 0.539160047 | 0.53906691     |
| Micro F1 | 0.637834969 | 0.637967631    |

### Parameters Explanation

Naive Bayes and the EM algorithm do not have many parameters, but many parameters can affect the data distribution. Most parameters are defined in `settings.py`

* `INF`: the initial value of log-likelihood
* `EPS`: the value to avoid division by zero
* `max_vocab_size`: the maximum size of vocabulary (Smaller values can speed algorithms up)
* `train_ratio`: the ratio of training size to test size
* `label_ratios`: the ratios of labeled size to unlabeled size
* `times`: times that we split data in the same setting in order to the reduce random error
* `main_metric`: the metric used to stop in EM algorithms
* `soft_sim`: using similarities on all classes or the maximum similarity

In some NB_EM baselines, soft_pathscore=True means we don't provide the hierachical information to classifiers but models can learn this information by themselves. 
