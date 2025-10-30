import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import umap
import numpy as np

from pathlib import Path

def probability_distribution(probabilities:np.ndarray, path:Path) -> None:

    """
    Plot probability distribution from predicted probabilities
    """
    sns.set(style="whitegrid")

    # Create a figure
    plt.figure(figsize=(8, 6))

    # Plot histogram with KDE
    sns.histplot(probabilities, kde=True, color='red', alpha=0.6)

    # Customize the plot
    plt.title('Toxicity Distribution', fontsize=14, pad=10)
    plt.xlabel('Toxicity', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xlim(0, 1)  # Probabilities are between 0 and 1
    plt.grid(True, alpha=0.3)

    # Show the plot
    plt.savefig(path / 'distribution.png')


def latent_space_plot(embedding_matrix, labels, path: Path, palette='bright', maxlen:int=10000) -> None:

    """
    plot peptides in reduced space via UMAP with corresponding labels
    """

    if len(embedding_matrix) > maxlen:
        embedding_matrix = embedding_matrix[:maxlen]

    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = reducer.fit_transform(embedding_matrix)
    df_umap = pd.DataFrame(
        {
            'Label': labels,
            'UMAP-1': umap_result[:, 0],
            'UMAP-2': umap_result[:, 1]
        }
    )
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))

    # Create scatter plot
    scatter = sns.scatterplot(
        data=df_umap,
        x="UMAP-1",
        y="UMAP-2",
        hue="Label",           # Color by acp category
        palette=palette,      # Color palette (you can try "muted", "bright", etc.)
        # size=50,            # Point size
        alpha=0.5,            # Transparency
        edgecolor="black",    # Edge color for better point definition
        linewidth=0.5         # Edge width
    )

    # Customize the plot
    plt.title("sequences in reduced space via UMAP", fontsize=20, pad=20)
    plt.xlabel("UMAP-1", fontsize=20)
    plt.ylabel("UMAP-2", fontsize=20)

    # Adjust legend
    plt.legend(
        title="type",
        title_fontsize=14,
        fontsize=16,
        loc="best"
    )

    # Remove top and right spines
    sns.despine()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Show plot
    plt.savefig(path / 'latent_space.png')
    plt.close()

# def latent_space_plot(embedding_matrix, labels, path:Path, pallete='bright') -> None:

#     """
#     plot peptides in reduced space via tSNE with corresponding labels
#     """

#     tsne = TSNE(n_components=2,perplexity=30,random_state=42)
#     tsne_result = tsne.fit_transform(embedding_matrix)
#     df_tsne = pd.DataFrame(
#         {
#             'Label' : labels,
#             'TSNE-1':tsne_result[:,0],
#             'TSNE-2':tsne_result[:,1] 
#         }
#     )
#     sns.set_style("whitegrid")
#     plt.figure(figsize=(10, 8))

#     # Create scatter plot
#     scatter = sns.scatterplot(
#         data=df_tsne,
#         x="TSNE-1",
#         y="TSNE-2",
#         hue="Label",           # Color by acp category
#         palette=pallete,      # Color palette (you can try "muted", "bright", etc.)
#         #size=50,            # Point size
#         alpha=0.5,          # Transparency
#         edgecolor="black",  # Edge color for better point definition
#         linewidth=0.5       # Edge width
#     )

#     # Customize the plot
#     plt.title("sequences in reduced space via t-SNE", fontsize=20, pad=20)
#     plt.xlabel("t-SNE-1", fontsize=20)
#     plt.ylabel("t-SNE-2", fontsize=20)

#     # Adjust legend
#     plt.legend(
#         title="type",
#         title_fontsize=14,
#         fontsize=16,
#         loc="best"
#     )

#     # Remove top and right spines
#     sns.despine()

#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()

#     # Show plot
#     plt.savefig(path / 'latent_space_tsne.png')

