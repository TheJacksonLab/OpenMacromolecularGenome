import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rdkit import Chem
from rdkit.Chem import AllChem

from sklearn.decomposition import PCA


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


if __name__ == '__main__':
    # save directory
    save_directory = '/home/sk77/PycharmProjects/publish/OMG/data/figure'

    # (1) load PolyInfo polymers
    df_PolyInfo = pd.read_csv(os.path.join(save_directory, '/home/sk77/PycharmProjects/publish/OMG/data/PolyInfo_SMILES_string.csv'))

    # canonical smiles
    df_PolyInfo['mol'] = df_PolyInfo['SMILES'].apply(lambda x: Chem.MolFromSmiles(x))
    df_PolyInfo['smiles'] = df_PolyInfo['mol'].apply(lambda x: Chem.MolToSmiles(x))
    df_PolyInfo = df_PolyInfo.drop(['SMILES'], axis=1)
    df_PolyInfo = df_PolyInfo.drop_duplicates(subset='smiles')

    print(df_PolyInfo.shape, flush=True)
    print(df_PolyInfo.head().to_string(), flush=True)

    # fps of PolyInfo polymers
    fps_PolyInfo = np.array(
        df_PolyInfo['mol'].apply(lambda x: AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024)).tolist()
    ).astype('float32')

    # (2) load OMG polymers (previous version)
    fps_OMG = np.load('/home/sk77/PycharmProjects/publish/OMG/data/fps_OMG_polymers_random_2M.npy')
    print(fps_OMG.shape, flush=True)

    # PCA
    n_components = 16
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(fps_PolyInfo)
    polyinfo_mean = np.mean(fps_PolyInfo, axis=0)

    principal_PolyInfo_df = pd.DataFrame(
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
    plt.savefig(os.path.join(save_directory, 'Explained_ratio_PolyInfo_polymers.png'))
    plt.show()
    plt.close()

    # (2) PCA of OMG polymers using PolyInfo polymers PCA
    principal_OMG_df = pca_using_common_axis(target_fps=fps_OMG, pca_eigenvector=pca.components_,
                                             mean_fps=polyinfo_mean)

    # construct dataframe for plot
    principal_PolyInfo_df['polymer'] = ['PolyInfo'] * principal_PolyInfo_df.shape[0]
    principal_OMG_df['polymer'] = ['OMG'] * principal_OMG_df.shape[0]

    sorted_principal_PolyInfo_df = principal_PolyInfo_df.sort_values(by='PC1', ascending=False)
    # print(sorted_principal_PolyInfo_df.head(50))

    plot_df = pd.concat([principal_PolyInfo_df, principal_OMG_df], ignore_index=True, axis=0)

    g = sns.jointplot(
        data=plot_df,
        x="PC1",
        y="PC2",
        hue="polymer",
        kind='scatter',
        joint_kws=dict(alpha=0.5),
        # hue_order=['PolyInfo', 'OMG'],
        hue_order=['OMG', 'PolyInfo'],
        edgecolor=None,
        palette=['c', 'm'],
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
    g.set_axis_labels('Principal Component 1', 'Principal Component 2', fontsize=12)
    g.fig.tight_layout()
    g.fig.savefig(os.path.join(save_directory, 'seaborn_PolyInfo_OMG_polymers_2M.png'), dpi=300)
