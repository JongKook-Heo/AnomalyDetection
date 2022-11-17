from itertools import product
from rdkit.Chem import AllChem, Descriptors
from rdkit import Chem
import pandas as pd
import numpy as np
import csv
from itertools import product


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

def read_moleculenet_smiles(data_path: str, target: str, task: str, print_info=True):
    smiles_data, labels, garbages = [], [], []
    data_name = data_path.split('/')[-1].split('.')[0].upper()
    with open(data_path) as f:
        csv_reader = csv.DictReader(f, delimiter=',')
        for idx, row in enumerate(csv_reader):
            smiles = row['smiles']
            label = row[target]
            mol = Chem.MolFromSmiles(smiles)

            if mol != None and label != '':
                smiles_data.append(smiles)
                if task == 'classification':
                    labels.append(int(label))
                elif task == 'regression':
                    labels.append(float(label))
                else:
                    raise ValueError('Task Error')

            elif mol is None:
                if print_info:
                    print(idx)
                garbages.append(smiles)
    if print_info:
        print(f'{data_name} | Target : {target}({task})| Total {len(smiles_data)}/{idx+1} instances')
    return smiles_data, labels, garbages

def smiles_to_df_with_fingerprint(smiles, label):
    df = pd.DataFrame()
    df['smiles'] = smiles
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    fingerprints = [list(AllChem.GetMorganFingerprintAsBitVect(mol, 2)) for mol in mols]
    df2 = pd.DataFrame(fingerprints, columns=[f'X{i:04d}' for i in range(len(fingerprints[0]))])
    df_total = pd.concat([df, df2], axis=1)
    df_total['label'] = label
    df_total['label'] = df_total['label'].apply(lambda x: -2 * x + 1)
    return df_total

def smiles_to_df_with_descriptors(smiles, label):
    df = pd.DataFrame()
    df['smiles'] = smiles
    features = []
    mols = [Chem.MolFromSmiles(m) for m in smiles]
    for feature in DESCRIPTORS:
        df[feature] = [getattr(Descriptors, feature)(m) for m in mols]
        features.append(feature)
        
    df['label'] = label
    df['label'] = df['label'].apply(lambda x: -2 * x + 1)
    df.fillna(df.mean(), inplace=True)
    return df

def split_by_label(df: pd.DataFrame):
    negatives = df[df['label'] == 1].drop(['smiles'], axis=1).reset_index(drop=True)
    positives = df[df['label'] == -1].drop(['smiles'], axis=1).reset_index(drop=True)
    
    neg_features = np.array(negatives.drop(['label'], axis=1))
    pos_features = np.array(positives.drop(['label'], axis=1))
    neg_labels = np.array(negatives['label'])
    pos_labels = np.array(positives['label'])
    return neg_features, pos_features, neg_labels, pos_labels

def comb_product(comb):
    return (dict(zip(comb.keys(), values)) for values in product(*comb.values()))
