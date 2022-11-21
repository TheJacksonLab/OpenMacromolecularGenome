import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.decomposition import PCA


def get_morgan_fingerprints(mol, radius=2, n_bits=1024):
    fp_array = np.zeros((1,), dtype=float)
    fp_vect = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    DataStructs.ConvertToNumpyArray(fp_vect, fp_array)

    return fp_array


def pca_using_common_axis(target_fps, pca_eigenvector, mean_fps):
    # data = target_fps - np.mean(target_fps, axis=0)
    data = target_fps - mean_fps
    # cov_matrix = data.T.dot(data) / data.shape[0]
    principal_df = pd.DataFrame(data=None, columns=['PC%d' % num for num in range(1, 3)])
    idx = 0
    for eigenvector in pca_eigenvector:
        # get PC coefficient
        idx += 1
        principal_df['PC%d' % idx] = data.dot(eigenvector)
        if idx == 2:
            break

    return principal_df


def check_isotope(mol):
    cnt = 0
    atom_data = [(atom, atom.GetIsotope()) for atom in mol.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            cnt += 1
            break
    return cnt


if __name__ == '__main__':
    save_directory = '/home/sk77/PycharmProjects/publish/OMG/data/figure'
    # PCA comparison with (1) eMolecules (2) QM9, (3) ZINC, and (4) OMG monomer reactants based on PolyInfo PCA
    # (1) load eMolecules - 250K random samples -> 249,961
    # df_eMolecules = pd.read_csv('version.smi', sep=' ')
    # df_eMolecules = df_eMolecules.rename(columns={'isosmiles': 'smiles'})
    # df_eMolecules = df_eMolecules.dropna(axis=0)
    #
    # # filter smiles containing '.'
    # df_eMolecules['point'] = df_eMolecules['smiles'].apply(lambda x: '.' in x)
    # df_eMolecules = df_eMolecules[~df_eMolecules['point']]
    #
    # # drop duplicates
    # df_eMolecules = df_eMolecules.drop_duplicates(subset=['smiles'])
    #
    # # random sampling: 250K
    # df_eMolecules = df_eMolecules.sample(n=250000, random_state=42)
    #
    # # convert to mol objects
    # df_eMolecules['mol'] = df_eMolecules['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    # df_eMolecules = df_eMolecules.dropna(axis=0)
    #
    # # canonical smiles - contain stereochemistry
    # df_eMolecules = df_eMolecules.drop(labels=['smiles'], axis=1)
    # df_eMolecules['smiles'] = df_eMolecules['mol'].apply(lambda x: Chem.MolToSmiles(x))
    # df_eMolecules = df_eMolecules.dropna(axis=0)
    #
    # # drop duplicates
    # df_eMolecules = df_eMolecules.drop_duplicates(subset=['smiles'])
    #
    # print(df_eMolecules.shape)
    # print(df_eMolecules.head().to_string())

    # (2) QM9 - all monomers -> 133,885
    df_QM9 = pd.read_csv('/home/sk77/PycharmProjects/publish/OMG/data/qm9_monomers.csv').drop(['Unnamed: 0'], axis=1)
    df_QM9 = df_QM9.rename(columns={'smile': 'smiles'})

    # convert to mol objects
    df_QM9['mol'] = df_QM9['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df_QM9 = df_QM9.dropna(axis=0)
    df_QM9 = df_QM9.drop(labels=['smiles'], axis=1)
    df_QM9['smiles'] = df_QM9['mol'].apply(lambda x: Chem.MolToSmiles(x))

    print(df_QM9.shape)
    print(df_QM9.head().to_string())

    # (3) ZINC - 250K -> 249,696
    df_ZINC = pd.read_csv('/home/sk77/PycharmProjects/publish/OMG/data/zinc_monomers_250K.csv').drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    df_ZINC = df_ZINC.rename(columns={'smile': 'smiles'})

    # convert to mol objects
    df_ZINC['mol'] = df_ZINC['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    df_ZINC = df_ZINC.dropna(axis=0)
    df_ZINC = df_ZINC.drop(labels=['smiles'], axis=1)
    df_ZINC['smiles'] = df_ZINC['mol'].apply(lambda x: Chem.MolToSmiles(x))

    print(df_ZINC.shape)
    print(df_ZINC.head().to_string())

    # (4) OMG monomer reactants - get OMG monomer reactants - 77,281
    df_OMG_monomers = pd.read_csv('/home/sk77/PycharmProjects/publish/OMG/data/OMG_monomers.csv')
    df_OMG_monomers['mol'] = df_OMG_monomers['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

    # PolyInfo monomers for PCA -> total 14,167 monomers (additional functional groups)
    df_PolyInfo_monomers = pd.read_csv('/home/sk77/PycharmProjects/publish/OMG/data/polyinfo_monomers.csv')
    df_PolyInfo_monomers['mol'] = df_PolyInfo_monomers['smiles'].apply(lambda x: Chem.MolFromSmiles(x))

    # PCA using fingerprints
    # get fingerprints
    # (1) eMolecules
    # fps_eMolecules = np.array(
    #     df_eMolecules['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)).tolist()
    # ).astype('float32')

    # (2) QM9
    fps_QM9 = np.array(
        df_QM9['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)).tolist()
    ).astype('float32')

    # (3) ZINC
    fps_ZINC = np.array(
        df_ZINC['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)).tolist()
    ).astype('float32')

    # (4) OMG monomers
    fps_OMG_monomers = np.array(
        df_OMG_monomers['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)).tolist()
    ).astype('float32')

    # (5) PolyInfo monomers
    fps_PolyInfo_monomers = np.array(
        df_PolyInfo_monomers['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)).tolist()
    ).astype('float32')

    # PCA
    n_components = 16
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(fps_PolyInfo_monomers)
    polyinfo_mean = np.mean(fps_PolyInfo_monomers, axis=0)

    principal_PolyInfo_monomers_df = pd.DataFrame(
        data=principal_components,
        columns=['PC%d' % (num + 1) for num in range(n_components)]
    )

    # plot explained variance by principal component analysis
    exp_var_pca = pca.explained_variance_ratio_
    plt.bar(range(1, len(exp_var_pca) + 1), exp_var_pca, alpha=0.5, align='center',
            label='Individual explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(save_directory, 'Explained_ratio_PolyInfo_monomers.png'))
    plt.show()
    plt.close()

    # # (1) PCA of eMolecules using PolyInfo monomers PCA
    # principal_eMolecules_df = pca_using_common_axis(target_fps=fps_eMolecules, pca_eigenvector=pca.components_,
    #                                                 mean_fps=polyinfo_mean)

    # (2) PCA of QM9 using PolyInfo monomers PCA
    principal_QM9_df = pca_using_common_axis(target_fps=fps_QM9, pca_eigenvector=pca.components_,
                                             mean_fps=polyinfo_mean)

    # (3) PCA of ZINC using PolyInfo monomers PCA
    principal_ZINC_df = pca_using_common_axis(target_fps=fps_ZINC, pca_eigenvector=pca.components_,
                                              mean_fps=polyinfo_mean)

    # (4) PCA of OMG monomer reactants using PolyInfo monomers PCA
    principal_OMG_monomers_df = pca_using_common_axis(target_fps=fps_OMG_monomers, pca_eigenvector=pca.components_,
                                                      mean_fps=polyinfo_mean)

    # get min and max
    # print(principal_ZINC_df['PC1'].describe())
    # print(principal_ZINC_df['PC2'].describe())
    # print(principal_OMG_monomers_df['PC1'].describe())
    # print(principal_OMG_monomers_df['PC2'].describe())
    # print(principal_QM9_df['PC1'].describe())
    # print(principal_QM9_df['PC2'].describe())
    # exit()

    # construct dataframe for plot
    # principal_eMolecules_df['monomer'] = ['eMolecules'] * principal_eMolecules_df.shape[0]
    principal_QM9_df['monomer'] = ['QM9'] * principal_QM9_df.shape[0]
    principal_PolyInfo_monomers_df['monomer'] = ['PolyInfo'] * principal_PolyInfo_monomers_df.shape[0]
    principal_ZINC_df['monomer'] = ['ZINC'] * principal_ZINC_df.shape[0]
    principal_OMG_monomers_df['monomer'] = ['OMG'] * principal_OMG_monomers_df.shape[0]

    # seaborn plot
    # plot_df = pd.concat([principal_eMolecules_df, principal_PolyInfo_monomers_df], ignore_index=True, axis=0)
    # plot_df = pd.concat([principal_OMG_monomers_df, principal_PolyInfo_monomers_df], ignore_index=True, axis=0)
    plot_df = pd.concat([principal_ZINC_df, principal_OMG_monomers_df, principal_QM9_df], ignore_index=True, axis=0)
    # plot_df = pd.concat([principal_PolyInfo_monomers_df, principal_QM9_df], ignore_index=True, axis=0)
    # plot_df = pd.concat([principal_eMolecules_df], ignore_index=True, axis=0)
    # plot_df = pd.concat([principal_PolyInfo_monomers_df], ignore_index=True, axis=0)
    g = sns.jointplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="monomer",
        kind='scatter',
        joint_kws=dict(alpha=0.5),
        # hue_order=['OMG', 'PolyInfo'],
        hue_order=['OMG', 'ZINC', 'QM9'],
        edgecolor=None,
        palette=['c', 'b', 'r'],
        # palette=['c', 'm'],
        s=5.0,
        marginal_kws={'common_norm': False}
    )
    g.ax_joint.tick_params(labelsize=12)
    legend = g.ax_joint.legend(title=None, fontsize=10, loc='upper right')
    legend.get_frame().set_edgecolor('k')
    legend.get_frame().set_linewidth(1.0)
    for lh in g.ax_joint.legend_.legendHandles:
        lh.set_alpha(0.5)
        lh.set_sizes([5])
    # g.set_axis_labels('PC1', 'PC2', fontsize=12)
    g.set_axis_labels('Principal Component 1', 'Principal Component 2', fontsize=12)
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(save_directory, "seaborn_OMG_ZINC_QM9.png"), dpi=300)
    # g.fig.savefig(os.path.join(save_directory, "seaborn_OMG_PolyInfo_black.png"), dpi=300)

