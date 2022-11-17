# Anomaly Detection Tutorial for MUV Dataset
#### 제작 : 허종국 (hjkso1406@korea.ac.kr)

## Introduction
### Moleculenet Benchmark
뇌혈관장벽 투과성, 용해도, 전기음성도 등 화학 분자의 물성을 예측하는 것은 화학 정보학 분야에서 가장 중요한 태스크 중 하나입니다. [MoleculeNet Benchmark](https://pubs.rsc.org/en/content/articlehtml/2018/sc/c7sc02664a)는 뇌혈관장벽 투과성, 용해도, 전기음성도 등 양자역학, 물리화학, 생물물리학, 생리학에 아우르는 다양한 물성에 대한 데이터셋을 제공합니다.
![Moleculenet](./images/moleculenet.png)
출처 : Wu, Z., Ramsundar, B., Feinberg, E. N., Gomes, J., Geniesse, C., Pappu, A. S., ... & Pande, V. (2018). MoleculeNet: a benchmark for molecular machine learning. Chemical science, 9(2), 513-530.

### MUV Dataset
MoleculeNet Benchmark 의 데이터셋 중 하나인 MUV 데이터셋은 아래와 같이 총 8개의 타겟을 가진 이진 분류 데이터셋입니다. MUV는 간단히 말해 어떠한 촉매나 수용체에 반응하는 단백질을 검출하는 것입니다. 하지만 각각의 TARGET에 대한 클래스 불균형이 매우 심하여 __양성 비율__ 이 1퍼센트도 되지않는 경우가 많습니다.
![MUV](./images/muv.PNG)
출처 : https://jcheminf.biomedcentral.com/articles/10.1186/s13321-019-0407-y/tables/8

MUV 데이터셋에 대한 자세한 설명은 [논문 링크](https://pubs.acs.org/doi/pdf/10.1021/ci8002649)을 참조해주세요

### Purpose & Requirements
본 튜토리얼에서는 MUV Dataset에 대해 Classification 이 아닌 Anomaly Detection을 통해 양성 데이터를 검출하고자 합니다.

* Caution : 해당 튜토리얼을 진행하기에 앞서 rdkit 패키지를 설치해주세요! rdkit 패키지의 설치 명령어는 Python version에 따라 다릅니다.

__python 3.7이하__
```
pip install rdkit 
```
__python 3.8__
```
conda install -c conda-forge rdkit
```
* 본 튜토리얼과 동일한 가상환경으로 진행하고 싶다면 아래의 명령어를 실행해주세요
```
conda env create --file environment.yaml
```

### Download Data
본 튜토리얼에서 사용하는 데이터는 Moleculenet Benchmark의 MUV 데이터셋입니다. [링크](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view)를 통해 데이터를 다운받으시길 바랍니다. 혹은 `./data` 라는 폴더를 생성한 후 직접 [MoleculeNet](https://moleculenet.org/)에서 다운 받으실 수 있습니다.

## How to represent molecules??
분자 데이터를 표현하는 방식은 매우 다양합니다. 사용하고자 하는 모델에 따른 분자의 표현 방식은 다음과 같습니다.

### Molecular Representation(Structured Data)
1. __Molecular Descriptors__ : rdkit에서 제공하는 약 122가지의 Descriptor를 통해 분자의 특징을 추출하여 분자를 정형데이터로 변환합니다. Molecular Descriptor의 종류에는 __방향족 고리의 개수__ 나 __분자의 무게__ 등이 존재합니다. rdkit에서 제공하는 descriptor는 [링크](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors.html)를 참조하세요.

2. __Molecular Fingerprint__ : 분자를 __특정한 크기의 해시코드__ 로 변환합니다. 일반적으로 __Morganfingerprint__ 를 사용합니다. Morgan Fingerprint에 대한 자세한 설명은 [링크](https://www.youtube.com/watch?v=T2aHx4wVea8)를 참조해주세요.
![mr1](./images/mr1.PNG)

### Molecular Representation(Unstructured Data)
1. __SMILES__ : __언어모델__ 혹은 __1D CNN__ 을 사용할 때 [SMILES 표기 규칙](https://en.wikipedia.org/wiki/Simplified_molecular-input_line-entry_system)에 따라 작성된 문자열을 사용합니다.

2. __3D Graph Representation__ : __결합의 길이, 각도, 모든 원자 간의 거리__ 를 정보로 활용함으로써 __GNN__ 모델을 활용합니다. 하지만 이러한 모든 정보를 가진 데이터셋을 구축하는데에 많은 비용이 든다는 단점이 있습니다.

2. __2D Graph Representation__ : 원자를 node, 결합을 edge로 표현하는 그래프로 표현하면서, __GNN__ 계열의 모델을 활용합니다. 결합의 각도나 길이 등을 고려하지 못하여 [카이랄성 분자](https://ko.wikipedia.org/wiki/%EC%B9%B4%EC%9D%B4%EB%9E%84%EC%84%B1) 혹은 단일 결합의 회전으로 발생하는 [이성질체](https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%84%B1%EC%A7%88%EC%B2%B4) 를 잘 표현하지 못한다는 단점이 있습니다. 
![mr2](./images/mr2.PNG)


## OCSVM을 통한 분자데이터 이상치 탐지- Molecular Representation에 따른 정량적 비교
### Load Data
본 튜토리얼에서는 분자를 __정형 데이터__ 로 표현하는 두 가지 방식에 대해 이상치 탐지를 진행합니다. 정상과 이상을 구분 짓는 피쳐가 해당 표현 방법에서 존재하는지 알아보고, __어떠한 표현 방법이 더 유용한지__ 살펴보겠습니다. MUV 데이터셋의 여러가지 타겟 중 __MUV-692, MUV-689, MUV-846, MUV-859__ 만 사용합니다. 각 타겟별로 레이블이 없는 경우가 많기 때문에, 타겟 별로 별도의 데이터셋을 구축합니다.


```python
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Draw, AllChem, Descriptors
from rdkit import Chem
import csv
import warnings
import random
from utils import *
from collections import Counter
warnings.filterwarnings('ignore')


DESCRIPTORS = ['BCUT2D_CHGHI', 'BCUT2D_CHGLO', 'BCUT2D_LOGPHI', 'BCUT2D_LOGPLOW', 'BCUT2D_MRHI', 'BCUT2D_MRLOW',
               'BCUT2D_MWHI', 'BCUT2D_MWLOW', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n',
               'Chi1v', 'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'EState_VSA1', 'EState_VSA10',
               'EState_VSA11', 'EState_VSA2', 'EState_VSA3', 'EState_VSA4', 'EState_VSA5', 'EState_VSA6',
               'EState_VSA7', 'EState_VSA8', 'EState_VSA9', 'ExactMolWt', 'FpDensityMorgan1', 'FpDensityMorgan2',
               'FpDensityMorgan3', 'FractionCSP3', 'HallKierAlpha', 'HeavyAtomCount', 'HeavyAtomMolWt', 'Ipc',
               'Kappa1', 'Kappa2', 'Kappa3', 'LabuteASA', 'MaxAbsEStateIndex', 'MaxAbsPartialCharge', 'MaxEStateIndex',
               'MaxPartialCharge', 'MinAbsEStateIndex', 'MinAbsPartialCharge', 'MinEStateIndex', 'MinPartialCharge',
               'MolLogP', 'MolMR', 'MolWt', 'NHOHCount', 'NOCount', 'NumAliphaticCarbocycles', 'NumAliphaticHeterocycles',
               'NumAliphaticRings', 'NumAromaticCarbocycles', 'NumAromaticHeterocycles', 'NumAromaticRings', 'NumHAcceptors',
               'NumHDonors', 'NumHeteroatoms', 'NumRadicalElectrons', 'NumRotatableBonds', 'NumSaturatedCarbocycles',
               'NumSaturatedHeterocycles', 'NumSaturatedRings', 'NumValenceElectrons', 'PEOE_VSA1',
               'PEOE_VSA10', 'PEOE_VSA11', 'PEOE_VSA12', 'PEOE_VSA13', 'PEOE_VSA14', 'PEOE_VSA2', 'PEOE_VSA3', 'PEOE_VSA4',
               'PEOE_VSA5', 'PEOE_VSA6', 'PEOE_VSA7', 'PEOE_VSA8', 'PEOE_VSA9', 'RingCount', 'SMR_VSA1', 'SMR_VSA10', 'SMR_VSA2',
               'SMR_VSA3', 'SMR_VSA4', 'SMR_VSA5', 'SMR_VSA6', 'SMR_VSA7', 'SMR_VSA8', 'SMR_VSA9', 'SlogP_VSA1', 'SlogP_VSA10',
               'SlogP_VSA11', 'SlogP_VSA12', 'SlogP_VSA2', 'SlogP_VSA3', 'SlogP_VSA4', 'SlogP_VSA5', 'SlogP_VSA6', 'SlogP_VSA7',
               'SlogP_VSA8', 'SlogP_VSA9', 'TPSA', 'VSA_EState1', 'VSA_EState10', 'VSA_EState2', 'VSA_EState3', 'VSA_EState4',
               'VSA_EState5', 'VSA_EState6', 'VSA_EState7', 'VSA_EState8', 'VSA_EState9']


TARGET_LIST = ['MUV-692', 'MUV-689', 'MUV-846', 'MUV-859']

DATA_PATH = './data/muv/muv.csv'

datasets = dict()

num_samples, num_normals, num_abnormals = [], [], []
for target in TARGET_LIST:
    smiles, labels, _ = read_moleculenet_smiles(DATA_PATH, target, task='classification')
    num_samples.append(len(smiles))
    num_normals.append(Counter(labels)[0])
    num_abnormals.append(Counter(labels)[1])
    datasets[target] = {'MorganFingerprint': smiles_to_df_with_fingerprint(smiles, labels),
                        'Descriptors': smiles_to_df_with_descriptors(smiles, labels)}
```

    MUV | Target : MUV-692(classification)| Total 14647/93127 instances
    MUV | Target : MUV-689(classification)| Total 14606/93127 instances
    MUV | Target : MUV-846(classification)| Total 14714/93127 instances
    MUV | Target : MUV-859(classification)| Total 14751/93127 instances
    

아래 표에서 알 수 있듯이, 각 타겟 별로 정상 데이터는 약 14000개, 이상 데이터는 약 30개 정도로 극심한 클래스 불균형을 가지는 것을 알 수 있습니다.


```python
df_info = pd.DataFrame({'TARGET':TARGET_LIST,
                        'Size':num_samples,
                        '# of Normal': num_normals,
                        '# of Abnormals':num_abnormals})
print(df_info.to_markdown(index=False))
```

| TARGET   |   Size |   # of Normal |   # of Abnormals |
|:---------|-------:|--------------:|-----------------:|
| MUV-692  |  14647 |         14617 |               30 |
| MUV-689  |  14606 |         14577 |               29 |
| MUV-846  |  14714 |         14684 |               30 |
| MUV-859  |  14751 |         14727 |               24 |
    

### Molecular Representation with Descriptors
Molecular Descriptor로 분자를 표현할 경우 아래와 같이 나타낼 수 있습니다.


```python
print(datasets['MUV-692']['Descriptors'][['smiles', 'BCUT2D_CHGHI', 'NHOHCount', 'MolWt','VSA_EState3','label']].head().to_markdown(index=False))
```

    | smiles                                              |   BCUT2D_CHGHI |   NHOHCount |   MolWt |   VSA_EState3 |   label |
    |:----------------------------------------------------|---------------:|------------:|--------:|--------------:|--------:|
    | Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1                 |        2.06539 |           1 | 339.42  |      3.67898  |       1 |
    | NC(=O)NC(Cc1ccccc1)C(=O)O                           |        2.18126 |           4 | 208.217 |     10.9803   |       1 |
    | CCn1c(CSc2nccn2C)nc2cc(C(=O)O)ccc21                 |        2.12906 |           1 | 316.386 |     10.0182   |       1 |
    | CCCc1cc(=O)nc(SCC(=O)N(CC(C)C)C2CCS(=O)(=O)C2)[nH]1 |        2.34312 |           1 | 401.554 |      0.428556 |       1 |
    | Cc1cccc(NC(=O)N2CCC(c3nc4ccccc4o3)CC2)c1            |        2.26623 |           1 | 335.407 |      2.97648  |       1 |
    

### Molecular MorganFingerprint
Morgan Fingerprint로 분자를 표현할 경우 아래와 같이 나타낼 수 있습니다.


```python
print(datasets['MUV-692']['MorganFingerprint'][['smiles', 'X0000', 'X0001', 'X1024', 'X2047', 'label']].head().to_markdown(index=False))
```

    | smiles                                              |   X0000 |   X0001 |   X1024 |   X2047 |   label |
    |:----------------------------------------------------|--------:|--------:|--------:|--------:|--------:|
    | Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1                 |       0 |       0 |       0 |       0 |       1 |
    | NC(=O)NC(Cc1ccccc1)C(=O)O                           |       0 |       1 |       0 |       0 |       1 |
    | CCn1c(CSc2nccn2C)nc2cc(C(=O)O)ccc21                 |       0 |       0 |       0 |       0 |       1 |
    | CCCc1cc(=O)nc(SCC(=O)N(CC(C)C)C2CCS(=O)(=O)C2)[nH]1 |       0 |       1 |       0 |       0 |       1 |
    | Cc1cccc(NC(=O)N2CCC(c3nc4ccccc4o3)CC2)c1            |       0 |       0 |       0 |       0 |       1 |
    

### Visualize Samples ###
MUV 데이터셋은 __Virtual Screening__(특정 단백질 수용체 또는 효소에 결합할 가능성이 있는 분자를 식별)하기 위해 구축되었습니다. 그중 MUV-692는 [SF1 단백질](https://en.wikipedia.org/wiki/Steroidogenic_factor_1)에 반응하는 분자를 탐지하는 태스크입니다. 화학 및 생명과학에 대한 지식이 해박하진 않지만 눈으로 구분이 될지 모르니 양성(이상)과 음성(정상) 별로 3개씩 시각화 해보겠습니다.


```python
df_temp = datasets['MUV-692']['MorganFingerprint']
neg_sample = list(df_temp[df_temp['label'] == 1]['smiles'].sample(3))
pos_sample = list(df_temp[df_temp['label'] == -1]['smiles'].sample(3))
sample_smiles = neg_sample + pos_sample
sample_mols = [Chem.MolFromSmiles(m) for m in sample_smiles]
legends = ['normal'] * 3 + ['abnormal'] * 3
Draw.MolsToGridImage(sample_mols, legends=legends)
```




    
![png](AnomalyDetectionTutorial_files/AnomalyDetectionTutorial_11_0.png)
    



(결과적으로 눈으로는 구분할 수가 없네요...)  
Fingerprint나 Molecular Descriptor가 정상과 이상을 잘 구분하는 feature이길 기도합시다.

### Metric
정밀도, 위양성률, 정확도를 평가지표로 사용하였습니다.


```python
def metric(pred, target, return_value = True, print_info=True):
    assert pred.shape[0] == target.shape[0]
    
    TP_idx = np.where(np.logical_and(pred==-1, target==-1))[0]
    TP = len(TP_idx)
    FP_idx = np.where(np.logical_and(pred==-1, target==1))[0]
    FP = len(FP_idx)
    FN_idx = np.where(np.logical_and(pred==1, target==-1))[0]
    FN = len(FN_idx)
    TN_idx = np.where(np.logical_and(pred==1, target==1))[0]
    TN = len(TN_idx)
    
    s = pd.Series([None]*pred.shape[0])
    s.iloc[TP_idx] = 'TP'
    s.iloc[FP_idx] = 'FP'
    s.iloc[TN_idx] = 'TN'
    s.iloc[FN_idx] = 'FN'
    
    precision = TP/(TP + FP)
    fpr = FP/(FP + TN)
    acc = (TP + TN)/(TP + FP + FN +TN)
    if print_info:
        print(f'Precision : {recall:.2f}')
        print(f'FPR       : {fpr:.2f}')
        print(f'Accuracy  : {acc:.2f}')
        
    if return_value:
        return precision, fpr, acc, list(s)
```

### Experiment Setting
- 4개의 타겟에 대해, 분자의 표현 방식 별로 3가지 평가지표를 산출합니다.
- 정상 데이터 80%를 훈련 데이터, 정상 데이터 20%와 이상 데이터 전부를 테스트 데이터로 사용합니다.
- __각 표현 방식에 따른 특징 벡터가 서로 독립인지 파악하기 위해__ 주성분 분석을 통해 orthogonal한 주성분 벡터 20개로 차원 축소할 경우, 어떠한 차이가 나는지 비교하였습니다.
- kernel의 종류와 $\nu$ 에 따른 3가지 평가지표의 변화를 트래킹하였습니다.


```python
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from itertools import product
import time

def OCSVM_experiment(datasets: dict,
                     representation: str,
                     target: str,
                     seed: int=7,
                     kernel: str='poly',
                     use_pca: bool=False,
                     nu: int=0.5,
                     print_info: bool=False,
                     **kwargs):
    
    np.random.seed(seed)
    random.seed(seed)
    
    df = datasets[target][representation]
    
    neg_features, pos_features, neg_labels, pos_labels = split_by_label(df)
    
    X_train, neg_features_test, _, neg_labels_test = train_test_split(neg_features,
                                                                      neg_labels,
                                                                      test_size=0.2,
                                                                      random_state=seed) 
    X_test = np.concatenate([neg_features_test, pos_features], axis=0)
    y_test = np.concatenate([neg_labels_test, pos_labels])
    
    if representation == 'Descriptors':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif representation != 'MorganFingerprint':
        raise NotImplementedError
    
    variance = None
    if use_pca:
        pca = PCA(n_components=20)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)
        variance = pca.explained_variance_ratio_.sum() * 100.0

    s_time = time.time()
    ocsvm = OneClassSVM(kernel=kernel, nu=nu)
    ocsvm.fit(X_train)
    elapsed_time = time.time() - s_time
    
    pred = ocsvm.predict(X_test)
    precision, fpr, acc, _ = metric(pred, y_test, return_value=True, print_info=print_info)
    
    record = {'target':target,
              'representation':representation,
              'seed':seed,
              'kernel':kernel,
              'elapsed_time': elapsed_time,
              'use_pca':use_pca,
              'Variance Ratio': variance,
              'nu': nu, 
              'precision':precision,
              'fpr':fpr,
              'accuracy':acc}
    
    return record
```

## Result
타겟, 표현방식, 커널, 규제 파라미터, 차원 축소 여부에 따른 학습 시간과 평가지표를 산출하였습니다.  
### 차원 축소에 따른 변수 설명력 비교 ###
- Molecular Descriptor__ : 주성분 분석을 통해 전체 분산의 약 82%를 커버하였습니다.  
- Morgan Fingerprint__ : 주성분 20개를 합하여도 약 20%의 분산만을 설명할 수 있습니다.  

__Morgan Fingerprint 보다 Molecular Descriptor 표현 방식이 정보의 중복이 더욱 많은 것을 알 수 있습니다.__  
__가설 1__: Fingerprint 표현 방식의 경우 변수간 상관성이 더 작기 때문에, 모델 구축을 하기 위해서는 더 많은 변수를 사용해야 할 것이라 추측하였습니다. 따라서 고정된 개수로 차원 축소를 할 경우, Fingerprint 표현 방식에 대한 모델 설명력이 감소할 것이라 예상하였습니다.  
__검증 방식__ : 차원 축소할 경우 Descriptor방식보다 Fingerprint 표현 방식의 성능 하락이 더욱 클 것이다.


```python
from itertools import product
def comb_product(comb):
    return (dict(zip(comb.keys(), values)) for values in product(*comb.values()))

settings = {'datasets': [datasets],
            'seed': [777],
            'target': TARGET_LIST,
            'kernel':['poly', 'rbf'],
            'use_pca':[True, False],
            'nu':[0.2, 0.5, 0.8],
            'representation': ['Descriptors', 'MorganFingerprint']}

combinations = comb_product(settings)
records = []
for param in combinations:
    record = OCSVM_experiment(**param)
    records.append(record)
    
ocsvm_result_df_demo = pd.DataFrame.from_records(records)
print(ocsvm_result_df_demo.to_markdown(index=False))
```

    | target   | representation    |   seed | kernel   |   elapsed_time | use_pca   |   Variance Ratio |   nu |   precision |      fpr |   accuracy |
    |:---------|:------------------|-------:|:---------|---------------:|:----------|-----------------:|-----:|------------:|---------:|-----------:|
    | MUV-692  | Descriptors       |    777 | poly     |       1.5226   | True      |          82.3551 |  0.2 |  0.0130435  | 0.2329   |   0.762356 |
    | MUV-692  | MorganFingerprint |    777 | poly     |       0.960151 | True      |          27.5064 |  0.2 |  0.00984529 | 0.240766 |   0.753893 |
    | MUV-692  | Descriptors       |    777 | poly     |       2.43579  | True      |          82.3551 |  0.5 |  0.0109819  | 0.523598 |   0.477319 |
    | MUV-692  | MorganFingerprint |    777 | poly     |       2.15383  | True      |          27.5064 |  0.5 |  0.00822264 | 0.536252 |   0.463439 |
    | MUV-692  | Descriptors       |    777 | poly     |       2.33142  | True      |          82.3551 |  0.8 |  0.0103907  | 0.814295 |   0.192282 |
    | MUV-692  | MorganFingerprint |    777 | poly     |       2.22403  | True      |          27.5064 |  0.8 |  0.0108469  | 0.810876 |   0.196005 |
    | MUV-692  | Descriptors       |    777 | poly     |       2.23309  | False     |         nan      |  0.2 |  0.012285   | 0.274966 |   0.721056 |
    | MUV-692  | MorganFingerprint |    777 | poly     |      38.4853   | False     |         nan      |  0.2 |  0.0114566  | 0.206566 |   0.787745 |
    | MUV-692  | Descriptors       |    777 | poly     |       4.49569  | False     |         nan      |  0.5 |  0.0108418  | 0.530438 |   0.470548 |
    | MUV-692  | MorganFingerprint |    777 | poly     |      97.7404   | False     |         nan      |  0.5 |  0.0129428  | 0.495554 |   0.505755 |
    | MUV-692  | Descriptors       |    777 | poly     |       4.5534   | False     |         nan      |  0.8 |  0.0103778  | 0.815321 |   0.191266 |
    | MUV-692  | MorganFingerprint |    777 | poly     |     111.545    | False     |         nan      |  0.8 |  0.0121339  | 0.807456 |   0.200406 |
    | MUV-692  | Descriptors       |    777 | rbf      |       1.07127  | True      |          82.3551 |  0.2 |  0.00536673 | 0.19015  |   0.80264  |
    | MUV-692  | MorganFingerprint |    777 | rbf      |       1.05259  | True      |          27.5064 |  0.2 |  0.00680272 | 0.199726 |   0.7935   |
    | MUV-692  | Descriptors       |    777 | rbf      |       2.66432  | True      |          82.3551 |  0.5 |  0.00887978 | 0.496238 |   0.503047 |
    | MUV-692  | MorganFingerprint |    777 | rbf      |       2.70501  | True      |          27.5064 |  0.5 |  0.0117729  | 0.48803  |   0.512525 |
    | MUV-692  | Descriptors       |    777 | rbf      |       2.90899  | True      |          82.3551 |  0.8 |  0.0092827  | 0.80301  |   0.202437 |
    | MUV-692  | MorganFingerprint |    777 | rbf      |       2.80599  | True      |          27.5064 |  0.8 |  0.0106519  | 0.794118 |   0.212255 |
    | MUV-692  | Descriptors       |    777 | rbf      |       2.04099  | False     |         nan      |  0.2 |  0.00533808 | 0.191176 |   0.801625 |
    | MUV-692  | MorganFingerprint |    777 | rbf      |      37.9891   | False     |         nan      |  0.2 |  0.00664452 | 0.204514 |   0.788761 |
    | MUV-692  | Descriptors       |    777 | rbf      |       5.05094  | False     |         nan      |  0.5 |  0.00888585 | 0.495896 |   0.503385 |
    | MUV-692  | MorganFingerprint |    777 | rbf      |      96.546    | False     |         nan      |  0.5 |  0.00927152 | 0.511628 |   0.488152 |
    | MUV-692  | Descriptors       |    777 | rbf      |       5.92161  | False     |         nan      |  0.8 |  0.00967199 | 0.805404 |   0.200406 |
    | MUV-692  | MorganFingerprint |    777 | rbf      |     109.308    | False     |         nan      |  0.8 |  0.0104822  | 0.807114 |   0.199391 |
    | MUV-689  | Descriptors       |    777 | poly     |       1.5195   | True      |          82.8267 |  0.2 |  0.0119225  | 0.227366 |   0.767742 |
    | MUV-689  | MorganFingerprint |    777 | poly     |       1.046    | True      |          27.0683 |  0.2 |  0.00924499 | 0.220508 |   0.773854 |
    | MUV-689  | Descriptors       |    777 | poly     |       2.448    | True      |          82.8267 |  0.5 |  0.0101764  | 0.500343 |   0.49983  |
    | MUV-689  | MorganFingerprint |    777 | poly     |       2.2105   | True      |          27.0683 |  0.5 |  0.00905563 | 0.525377 |   0.474703 |
    | MUV-689  | Descriptors       |    777 | poly     |       2.4365   | True      |          82.8267 |  0.8 |  0.0101095  | 0.805898 |   0.20034  |
    | MUV-689  | MorganFingerprint |    777 | poly     |       2.34903  | True      |          27.0683 |  0.8 |  0.010906   | 0.808642 |   0.198302 |
    | MUV-689  | Descriptors       |    777 | poly     |       2.25301  | False     |         nan      |  0.2 |  0.00913838 | 0.260288 |   0.734805 |
    | MUV-689  | MorganFingerprint |    777 | poly     |      39.2782   | False     |         nan      |  0.2 |  0.0109718  | 0.216392 |   0.778268 |
    | MUV-689  | Descriptors       |    777 | poly     |       4.513    | False     |         nan      |  0.5 |  0.011471   | 0.502401 |   0.498472 |
    | MUV-689  | MorganFingerprint |    777 | poly     |      99.8395   | False     |         nan      |  0.5 |  0.0118655  | 0.51406  |   0.487267 |
    | MUV-689  | Descriptors       |    777 | poly     |       4.54379  | False     |         nan      |  0.8 |  0.0101911  | 0.799383 |   0.206791 |
    | MUV-689  | MorganFingerprint |    777 | poly     |     110.988    | False     |         nan      |  0.8 |  0.0113065  | 0.809671 |   0.197623 |
    | MUV-689  | Descriptors       |    777 | rbf      |       1.075    | True      |          82.8267 |  0.2 |  0.00811688 | 0.209534 |   0.78438  |
    | MUV-689  | MorganFingerprint |    777 | rbf      |       1.09102  | True      |          27.0683 |  0.2 |  0.0070922  | 0.192044 |   0.801358 |
    | MUV-689  | Descriptors       |    777 | rbf      |       2.7505   | True      |          82.8267 |  0.5 |  0.00924092 | 0.514746 |   0.485229 |
    | MUV-689  | MorganFingerprint |    777 | rbf      |       2.74679  | True      |          27.0683 |  0.5 |  0.00780696 | 0.479424 |   0.519185 |
    | MUV-689  | Descriptors       |    777 | rbf      |       2.90799  | True      |          82.8267 |  0.8 |  0.0101652  | 0.80144  |   0.204754 |
    | MUV-689  | MorganFingerprint |    777 | rbf      |       2.96304  | True      |          27.0683 |  0.8 |  0.0100503  | 0.8107   |   0.195586 |
    | MUV-689  | Descriptors       |    777 | rbf      |       2.016    | False     |         nan      |  0.2 |  0.0112     | 0.211934 |   0.782683 |
    | MUV-689  | MorganFingerprint |    777 | rbf      |      38.0882   | False     |         nan      |  0.2 |  0.0150502  | 0.201989 |   0.793209 |
    | MUV-689  | Descriptors       |    777 | rbf      |       5.32327  | False     |         nan      |  0.5 |  0.00981675 | 0.518861 |   0.481494 |
    | MUV-689  | MorganFingerprint |    777 | rbf      |      97.0271   | False     |         nan      |  0.5 |  0.0132363  | 0.511317 |   0.490662 |
    | MUV-689  | Descriptors       |    777 | rbf      |       5.56994  | False     |         nan      |  0.8 |  0.0105086  | 0.80727  |   0.199321 |
    | MUV-689  | MorganFingerprint |    777 | rbf      |     111.835    | False     |         nan      |  0.8 |  0.011078   | 0.795953 |   0.210866 |
    | MUV-846  | Descriptors       |    777 | poly     |       1.732    | True      |          81.7596 |  0.2 |  0.00877193 | 0.230848 |   0.763397 |
    | MUV-846  | MorganFingerprint |    777 | poly     |       0.935028 | True      |          28.0065 |  0.2 |  0.00848656 | 0.238679 |   0.755645 |
    | MUV-846  | Descriptors       |    777 | poly     |       2.496    | True      |          81.7596 |  0.5 |  0.00855826 | 0.512768 |   0.486687 |
    | MUV-846  | MorganFingerprint |    777 | poly     |       2.28516  | True      |          28.0065 |  0.5 |  0.0109044  | 0.525026 |   0.475902 |
    | MUV-846  | Descriptors       |    777 | poly     |       2.36801  | True      |          81.7596 |  0.8 |  0.0119098  | 0.790943 |   0.21638  |
    | MUV-846  | MorganFingerprint |    777 | poly     |       2.33599  | True      |          28.0065 |  0.8 |  0.0102881  | 0.818863 |   0.187732 |
    | MUV-846  | Descriptors       |    777 | poly     |       2.34183  | False     |         nan      |  0.2 |  0.0130208  | 0.258086 |   0.737782 |
    | MUV-846  | MorganFingerprint |    777 | poly     |      39.621    | False     |         nan      |  0.2 |  0.0101744  | 0.231869 |   0.762723 |
    | MUV-846  | Descriptors       |    777 | poly     |       5.09905  | False     |         nan      |  0.5 |  0.00968367 | 0.522302 |   0.477924 |
    | MUV-846  | MorganFingerprint |    777 | poly     |     106.307    | False     |         nan      |  0.5 |  0.0116732  | 0.518897 |   0.482305 |
    | MUV-846  | Descriptors       |    777 | poly     |       5.1775   | False     |         nan      |  0.8 |  0.011745   | 0.802179 |   0.205258 |
    | MUV-846  | MorganFingerprint |    777 | poly     |     114.659    | False     |         nan      |  0.8 |  0.0112688  | 0.806605 |   0.200539 |
    | MUV-846  | Descriptors       |    777 | rbf      |       1.068    | True      |          81.7596 |  0.2 |  0.0045045  | 0.225741 |   0.767442 |
    | MUV-846  | MorganFingerprint |    777 | rbf      |       1.113    | True      |          28.0065 |  0.2 |  0.00738007 | 0.18318  |   0.809909 |
    | MUV-846  | Descriptors       |    777 | rbf      |       2.7635   | True      |          81.7596 |  0.5 |  0.00738255 | 0.503575 |   0.495113 |
    | MUV-846  | MorganFingerprint |    777 | rbf      |       2.85953  | True      |          28.0065 |  0.5 |  0.00957592 | 0.49302  |   0.506572 |
    | MUV-846  | Descriptors       |    777 | rbf      |       2.88699  | True      |          81.7596 |  0.8 |  0.00890208 | 0.79605  |   0.208965 |
    | MUV-846  | MorganFingerprint |    777 | rbf      |       2.88663  | True      |          28.0065 |  0.8 |  0.010661   | 0.789922 |   0.21638  |
    | MUV-846  | Descriptors       |    777 | rbf      |       2.11099  | False     |         nan      |  0.2 |  0.00624025 | 0.216888 |   0.776542 |
    | MUV-846  | MorganFingerprint |    777 | rbf      |      38.3992   | False     |         nan      |  0.2 |  0.00169205 | 0.200885 |   0.791372 |
    | MUV-846  | Descriptors       |    777 | rbf      |       5.1516   | False     |         nan      |  0.5 |  0.00808625 | 0.501192 |   0.497809 |
    | MUV-846  | MorganFingerprint |    777 | rbf      |      98.9613   | False     |         nan      |  0.5 |  0.00684932 | 0.493701 |   0.50455  |
    | MUV-846  | Descriptors       |    777 | rbf      |       5.57305  | False     |         nan      |  0.8 |  0.00921659 | 0.805243 |   0.200202 |
    | MUV-846  | MorganFingerprint |    777 | rbf      |     113.799    | False     |         nan      |  0.8 |  0.00932994 | 0.795369 |   0.209976 |
    | MUV-859  | Descriptors       |    777 | poly     |       1.56     | True      |          82.8193 |  0.2 |  0.00570613 | 0.236592 |   0.758586 |
    | MUV-859  | MorganFingerprint |    777 | poly     |       1.0385   | True      |          28.4816 |  0.2 |  0.01443    | 0.23184  |   0.76532  |
    | MUV-859  | Descriptors       |    777 | poly     |       2.36526  | True      |          82.8193 |  0.5 |  0.00534045 | 0.505771 |   0.492929 |
    | MUV-859  | MorganFingerprint |    777 | poly     |       2.35252  | True      |          28.4816 |  0.5 |  0.010665   | 0.535302 |   0.466667 |
    | MUV-859  | Descriptors       |    777 | poly     |       2.44317  | True      |          82.8193 |  0.8 |  0.00666667 | 0.809233 |   0.194613 |
    | MUV-859  | MorganFingerprint |    777 | poly     |       2.37748  | True      |          28.4816 |  0.8 |  0.0097166  | 0.830278 |   0.176431 |
    | MUV-859  | Descriptors       |    777 | poly     |       2.3405   | False     |         nan      |  0.2 |  0.003861   | 0.262729 |   0.732323 |
    | MUV-859  | MorganFingerprint |    777 | poly     |      39.9803   | False     |         nan      |  0.2 |  0.00825083 | 0.204005 |   0.791246 |
    | MUV-859  | Descriptors       |    777 | poly     |       4.45402  | False     |         nan      |  0.5 |  0.00580271 | 0.523422 |   0.475758 |
    | MUV-859  | MorganFingerprint |    777 | poly     |     102.56     | False     |         nan      |  0.5 |  0.0046729  | 0.50611  |   0.492256 |
    | MUV-859  | Descriptors       |    777 | poly     |       4.8277   | False     |         nan      |  0.8 |  0.00790678 | 0.809233 |   0.195623 |
    | MUV-859  | MorganFingerprint |    777 | poly     |     113.881    | False     |         nan      |  0.8 |  0.00692341 | 0.779022 |   0.224579 |
    | MUV-859  | Descriptors       |    777 | rbf      |       1.107    | True      |          82.8193 |  0.2 |  0.00546448 | 0.185336 |   0.809091 |
    | MUV-859  | MorganFingerprint |    777 | rbf      |       1.08983  | True      |          28.4816 |  0.2 |  0.00701754 | 0.192125 |   0.802694 |
    | MUV-859  | Descriptors       |    777 | rbf      |       2.7685   | True      |          82.8193 |  0.5 |  0.0115962  | 0.491853 |   0.509764 |
    | MUV-859  | MorganFingerprint |    777 | rbf      |       2.75699  | True      |          28.4816 |  0.5 |  0.00889193 | 0.491853 |   0.508418 |
    | MUV-859  | Descriptors       |    777 | rbf      |       3.00599  | True      |          82.8193 |  0.8 |  0.00915141 | 0.808554 |   0.197306 |
    | MUV-859  | MorganFingerprint |    777 | rbf      |       2.88594  | True      |          28.4816 |  0.8 |  0.0088645  | 0.797013 |   0.208418 |
    | MUV-859  | Descriptors       |    777 | rbf      |       2.1865   | False     |         nan      |  0.2 |  0.0103806  | 0.194162 |   0.801347 |
    | MUV-859  | MorganFingerprint |    777 | rbf      |      39.3244   | False     |         nan      |  0.2 |  0.0106383  | 0.189409 |   0.806061 |
    | MUV-859  | Descriptors       |    777 | rbf      |       5.38316  | False     |         nan      |  0.5 |  0.0122034  | 0.494569 |   0.507407 |
    | MUV-859  | MorganFingerprint |    777 | rbf      |      99.8374   | False     |         nan      |  0.5 |  0.00949796 | 0.495587 |   0.505051 |
    | MUV-859  | Descriptors       |    777 | rbf      |       5.96355  | False     |         nan      |  0.8 |  0.00868486 | 0.813646 |   0.191919 |
    | MUV-859  | MorganFingerprint |    777 | rbf      |     116.3      | False     |         nan      |  0.8 |  0.00882353 | 0.800747 |   0.204714 |
    

### 학습 시간에 대한 비교 ###
차원 축소하지 않은 MorganFingerprint 표현 방식의 경우 학습 시간이 다른 조합 대비 40~80배 느린 것을 알 수 있습니다.

- 만약 해당 조합과 나머지 조합에 대한 평가 지표의 차이가 크지 않을 경우, __MorganFingerprint를 사용하는 것은 비효율적__ 일 것입니다.


```python
print(ocsvm_result_df_demo.groupby(['representation', 'use_pca', 'kernel'])['elapsed_time'].agg(['mean', 'std']).reset_index().to_markdown())
```

    |    | representation    | use_pca   | kernel   |     mean |       std |
    |---:|:------------------|:----------|:---------|---------:|----------:|
    |  0 | Descriptors       | False     | poly     |  3.90271 |  1.21207  |
    |  1 | Descriptors       | False     | rbf      |  4.35763 |  1.69701  |
    |  2 | Descriptors       | True      | poly     |  2.13819 |  0.415311 |
    |  3 | Descriptors       | True      | rbf      |  2.24817 |  0.86722  |
    |  4 | MorganFingerprint | False     | poly     | 84.5737  | 33.8131   |
    |  5 | MorganFingerprint | False     | rbf      | 83.1178  | 33.6279   |
    |  6 | MorganFingerprint | True      | poly     |  1.85569 |  0.639679 |
    |  7 | MorganFingerprint | True      | rbf      |  2.24636 |  0.859456 |
    

### 조합에 대한 평가 지표 비교 ###
4개의 타겟에 대한 평균과 표준편차를 구하여 조합별로 비교하였습니다


```python
summary = ocsvm_result_df_demo.groupby(['representation', 'use_pca', 'nu', 'kernel'])['precision', 'fpr', 'accuracy'].agg(['mean', 'std']).reset_index()
summary
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>representation</th>
      <th>use_pca</th>
      <th>nu</th>
      <th>kernel</th>
      <th colspan="2" halign="left">precision</th>
      <th colspan="2" halign="left">fpr</th>
      <th colspan="2" halign="left">accuracy</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Descriptors</td>
      <td>False</td>
      <td>0.2</td>
      <td>poly</td>
      <td>0.009576</td>
      <td>0.004166</td>
      <td>0.264017</td>
      <td>0.007541</td>
      <td>0.731492</td>
      <td>0.007306</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Descriptors</td>
      <td>False</td>
      <td>0.2</td>
      <td>rbf</td>
      <td>0.008290</td>
      <td>0.002930</td>
      <td>0.203540</td>
      <td>0.012773</td>
      <td>0.790549</td>
      <td>0.012876</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Descriptors</td>
      <td>False</td>
      <td>0.5</td>
      <td>poly</td>
      <td>0.009450</td>
      <td>0.002542</td>
      <td>0.519640</td>
      <td>0.012044</td>
      <td>0.480675</td>
      <td>0.012261</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Descriptors</td>
      <td>False</td>
      <td>0.5</td>
      <td>rbf</td>
      <td>0.009748</td>
      <td>0.001783</td>
      <td>0.502630</td>
      <td>0.011193</td>
      <td>0.497524</td>
      <td>0.011388</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Descriptors</td>
      <td>False</td>
      <td>0.8</td>
      <td>poly</td>
      <td>0.010055</td>
      <td>0.001591</td>
      <td>0.806529</td>
      <td>0.007179</td>
      <td>0.199734</td>
      <td>0.007504</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Descriptors</td>
      <td>False</td>
      <td>0.8</td>
      <td>rbf</td>
      <td>0.009521</td>
      <td>0.000772</td>
      <td>0.807891</td>
      <td>0.003945</td>
      <td>0.197962</td>
      <td>0.004056</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Descriptors</td>
      <td>True</td>
      <td>0.2</td>
      <td>poly</td>
      <td>0.009861</td>
      <td>0.003308</td>
      <td>0.231927</td>
      <td>0.003859</td>
      <td>0.763020</td>
      <td>0.003766</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Descriptors</td>
      <td>True</td>
      <td>0.2</td>
      <td>rbf</td>
      <td>0.005863</td>
      <td>0.001563</td>
      <td>0.202690</td>
      <td>0.018588</td>
      <td>0.790888</td>
      <td>0.018811</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Descriptors</td>
      <td>True</td>
      <td>0.5</td>
      <td>poly</td>
      <td>0.008764</td>
      <td>0.002495</td>
      <td>0.510620</td>
      <td>0.010036</td>
      <td>0.489191</td>
      <td>0.009564</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Descriptors</td>
      <td>True</td>
      <td>0.5</td>
      <td>rbf</td>
      <td>0.009275</td>
      <td>0.001744</td>
      <td>0.501603</td>
      <td>0.010008</td>
      <td>0.498288</td>
      <td>0.010567</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Descriptors</td>
      <td>True</td>
      <td>0.8</td>
      <td>poly</td>
      <td>0.009769</td>
      <td>0.002214</td>
      <td>0.805092</td>
      <td>0.010045</td>
      <td>0.200904</td>
      <td>0.010859</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Descriptors</td>
      <td>True</td>
      <td>0.8</td>
      <td>rbf</td>
      <td>0.009375</td>
      <td>0.000550</td>
      <td>0.802264</td>
      <td>0.005145</td>
      <td>0.203366</td>
      <td>0.004860</td>
    </tr>
    <tr>
      <th>12</th>
      <td>MorganFingerprint</td>
      <td>False</td>
      <td>0.2</td>
      <td>poly</td>
      <td>0.010213</td>
      <td>0.001411</td>
      <td>0.214708</td>
      <td>0.012625</td>
      <td>0.779996</td>
      <td>0.012753</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MorganFingerprint</td>
      <td>False</td>
      <td>0.2</td>
      <td>rbf</td>
      <td>0.008506</td>
      <td>0.005694</td>
      <td>0.199200</td>
      <td>0.006701</td>
      <td>0.794851</td>
      <td>0.007693</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MorganFingerprint</td>
      <td>False</td>
      <td>0.5</td>
      <td>poly</td>
      <td>0.010289</td>
      <td>0.003785</td>
      <td>0.508655</td>
      <td>0.010202</td>
      <td>0.491896</td>
      <td>0.010093</td>
    </tr>
    <tr>
      <th>15</th>
      <td>MorganFingerprint</td>
      <td>False</td>
      <td>0.5</td>
      <td>rbf</td>
      <td>0.009714</td>
      <td>0.002637</td>
      <td>0.503058</td>
      <td>0.009747</td>
      <td>0.497104</td>
      <td>0.008949</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MorganFingerprint</td>
      <td>False</td>
      <td>0.8</td>
      <td>poly</td>
      <td>0.010408</td>
      <td>0.002357</td>
      <td>0.800689</td>
      <td>0.014502</td>
      <td>0.205787</td>
      <td>0.012600</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MorganFingerprint</td>
      <td>False</td>
      <td>0.8</td>
      <td>rbf</td>
      <td>0.009928</td>
      <td>0.001034</td>
      <td>0.799796</td>
      <td>0.005441</td>
      <td>0.206237</td>
      <td>0.005310</td>
    </tr>
    <tr>
      <th>18</th>
      <td>MorganFingerprint</td>
      <td>True</td>
      <td>0.2</td>
      <td>poly</td>
      <td>0.010502</td>
      <td>0.002677</td>
      <td>0.232948</td>
      <td>0.009128</td>
      <td>0.762178</td>
      <td>0.009265</td>
    </tr>
    <tr>
      <th>19</th>
      <td>MorganFingerprint</td>
      <td>True</td>
      <td>0.2</td>
      <td>rbf</td>
      <td>0.007073</td>
      <td>0.000239</td>
      <td>0.191769</td>
      <td>0.006765</td>
      <td>0.801865</td>
      <td>0.006724</td>
    </tr>
    <tr>
      <th>20</th>
      <td>MorganFingerprint</td>
      <td>True</td>
      <td>0.5</td>
      <td>poly</td>
      <td>0.009712</td>
      <td>0.001288</td>
      <td>0.530489</td>
      <td>0.006120</td>
      <td>0.470178</td>
      <td>0.006082</td>
    </tr>
    <tr>
      <th>21</th>
      <td>MorganFingerprint</td>
      <td>True</td>
      <td>0.5</td>
      <td>rbf</td>
      <td>0.009512</td>
      <td>0.001674</td>
      <td>0.488082</td>
      <td>0.006153</td>
      <td>0.511675</td>
      <td>0.005591</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MorganFingerprint</td>
      <td>True</td>
      <td>0.8</td>
      <td>poly</td>
      <td>0.010439</td>
      <td>0.000557</td>
      <td>0.817165</td>
      <td>0.009782</td>
      <td>0.189618</td>
      <td>0.009894</td>
    </tr>
    <tr>
      <th>23</th>
      <td>MorganFingerprint</td>
      <td>True</td>
      <td>0.8</td>
      <td>rbf</td>
      <td>0.010057</td>
      <td>0.000845</td>
      <td>0.797938</td>
      <td>0.008992</td>
      <td>0.208159</td>
      <td>0.008991</td>
    </tr>
  </tbody>
</table>
</div>



### 결과 요약 ###
$\nu$ 에 따른 정밀도, 위양성률, 정확도의 추이를 살펴보았습니다.
- $\nu$ 에 따른 각 평가지표의 변동성이 큰 걸로 보아, 정상 데이터와 이상 데이터의 영역이 잘 구분되지 않았다고 판단 할 수 있습니다. 따라서 __두 가지 표현 방식 모두가 MUV 데이터의 정상/이상을 판단하는데 도움이 되지 않는 것을 알 수 있습니다.__  
- MUV데이터에 대해 개별 표현 방식에 따른 표현력의 우위가 없는 것을 알 수 있습니다. precision의 경우 차이가 커보이지만, __y축 스케일이 매우 작은 것을 유의해서 봐야합니다.__  
- __가설1__(Fingerprint 방식의 경우 차원 축소로 인한 성능차이가 클 것)이 기각되었음을 알 수 있습니다. __개별 변수 간의 상관성은 없지만, 각 변수가 MUV의 타겟을 설명하는 설명력은 존재하지 않는다고 판단하였습니다.__


```python
keys = list(product(*[[True, False], ['poly', 'rbf']]))

fig, axs = plt.subplots(3, 4, figsize=(16, 12))

keys = list(product(*[[True, False], ['poly', 'rbf']]))
for j, k in enumerate(keys):
    for i, m in enumerate(['precision', 'fpr', 'accuracy']):
        
        use_pca, kernel = k
        
        
        summary_desc = summary[(summary['representation'] == 'Descriptors')&(summary['use_pca'] == use_pca)&(summary['kernel'] == kernel)][m]
        summary_fp = summary[(summary['representation'] == 'MorganFingerprint')&(summary['use_pca'] == use_pca)&(summary['kernel'] == kernel)][m]
        nu = list(summary['nu'].unique())
        desc_mean = list(summary_desc['mean'])
        desc_std = list(summary_desc['std'])
        
        fp_mean = list(summary_fp['mean'])
        fp_std = list(summary_fp['std'])
        
        use_pca = str(use_pca)
        subtitle = f'PCA:{use_pca}&kernel:{kernel}'
        
        axs[i, j].errorbar(nu, fp_mean, yerr=fp_std, label='fingeprint')
        axs[i, j].errorbar(nu, desc_mean, yerr=desc_std, label='descriptors')
        axs[i, j].legend()
        
        axs[i, j].set(xlabel=subtitle, ylabel=m)
        axs[i, j].label_outer()
fig.suptitle('Comparison', fontsize=15)
fig.tight_layout()
```


    
![png](AnomalyDetectionTutorial_files/AnomalyDetectionTutorial_24_0.png)
    


### 표현 방식에 따른 정상/이상 데이터 시각화 ###
각 플롯 별 왼쪽 그림은 학습(정상)데이터에 대한 결정 경계, 오른쪽 그림은 테스트 데이터(정상 + 이상)에 대한 결정 경계를 시각화하였습니다.
- 테스트 데이터에 대한 시각화를 통해 두 표현 방식에 사용되었던 변수는 MUV의 타겟 단백질을 구분할 설명력이 없다는 것을 알 수 있습니다.


```python
from mlxtend.plotting import plot_decision_regions
import matplotlib.pyplot as plt

def plot_ocsvm(dataset: dict, target:str, kernel:kernel, nu:int, representation='Descriptors'):
    df = datasets[target][representation]

    neg_features, pos_features, neg_labels, pos_labels = split_by_label(df)

    X_train, neg_features_test, _, neg_labels_test = train_test_split(neg_features,
                                                                    neg_labels,
                                                                    test_size=0.2,
                                                                    random_state=777) 
    X_test = np.concatenate([pos_features, neg_features_test], axis=0)
    y_test = np.concatenate([pos_labels, neg_labels_test])
    
    if representation == 'Descriptors':
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    elif representation != 'MorganFingerprint':
        raise NotImplementedError
    
    pca = PCA(n_components=2)
    
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    ocsvm = OneClassSVM(kernel='rbf', nu=nu)
    ocsvm.fit(X_train)
    pred = ocsvm.predict(X_test)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.title.set_text("Train Data")
    plot_decision_regions(X_train, np.ones(X_train.shape[0], dtype=int), clf=ocsvm, ax=ax1)
    
    ax2.title.set_text("Test Data")
    plot_decision_regions(X_test, y_test, clf=ocsvm, ax=ax2)
    fig.suptitle(f'Representation : {representation} & Kernel : {kernel}', fontsize=15)
    fig.tight_layout()

plot_ocsvm(datasets, 'MUV-846', 'poly', 0.02)
```


    
![png](AnomalyDetectionTutorial_files/AnomalyDetectionTutorial_26_0.png)
    



```python
plot_ocsvm(datasets, 'MUV-846', 'rbf', 0.02)
```


    
![png](AnomalyDetectionTutorial_files/AnomalyDetectionTutorial_27_0.png)
    



```python
plot_ocsvm(datasets, 'MUV-846', 'rbf', 0.02,'MorganFingerprint')
```


    
![png](AnomalyDetectionTutorial_files/AnomalyDetectionTutorial_28_0.png)
    



```python
plot_ocsvm(datasets, 'MUV-846', 'poly', 0.02,'MorganFingerprint')
```


    
![png](AnomalyDetectionTutorial_files/AnomalyDetectionTutorial_29_0.png)
    


## To Do - How to Improve??
본 튜토리얼은 분자를 표현하는 두가지 방식을 통해 OCSVM으로 이상 데이터를 검출하는 프로세스를 진행하였습니다. 하지만 아쉽게도, MUV 데이터셋의 타겟에 대해 두 가지 표현 방식은 좋은 설명력을 가지지 못하는 것을 확인할 수 있습니다.
- 두 표현 방식을 통한 입력 변수가 정상/이상 판단에 설명력이 없습니다. 따라서 입력 변수를 바꾸지 않는한 Isolation Forest 등의 다른 모델을 사용하여도 이상치 탐지가 잘 되지 않을 것이라 생각합니다.
- Morgan Fingerprint나 Molecular Descriptors와 같은 정형 데이터 방식이 아닌 __그래프 데이터__ 를 입력으로 하는 __GNN 기반 이상치 탐지__ 를 고려해볼 수 있습니다. 차후 분석에서는 분자를 그래프로 표현한 후, __Masked Node Prediction__ 을 통해 정상 데이터를 학습 시킨 후, 복원 오차를 통해 이상치를 탐지해볼 계획입니다.

