from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pickle
import os

# tsne_features_path = 'simple_autoe_tsne.pickle'
# autoe_features_path = 'simple_autoe_features.pickle'
# autoe_labels_path = 'simple_autoe_labels.pickle'

tsne_features_path = 'sparse_autoe_tsne.pickle'
autoe_features_path = 'sparse_autoe_features.pickle'
autoe_labels_path = 'sparse_autoe_labels.pickle'

# tsne_features_path = 'deep_autoe_tsne.pickle'
# autoe_features_path = 'deep_autoe_features.pickle'
# autoe_labels_path = 'deep_autoe_labels.pickle'

# TODO
# tsne_features_path = 'denoise_autoe_tsne.pickle'
# autoe_features_path = 'denoise_autoe_features.pickle'

# TODO
# tsne_features_path = 'conv_autoe_tsne.pickle'
# autoe_features_path = 'conv_autoe_features.pickle'

tsne_features = None

if os.path.exists(autoe_labels_path):
    labels = pickle.load(open(autoe_labels_path, 'rb'))

    if os.path.exists(tsne_features_path):
        print('t-sne features found. Loading ...')
        tsne_features = pickle.load(open(tsne_features_path, 'rb'))
    else:
        if os.path.exists(autoe_features_path):
            print('Pre-extracted features found. Loading them ...')
            latent_space = pickle.load(open(autoe_features_path, 'rb'))

            print('t-SNE happening ...!')
            tsne_features = TSNE().fit_transform(latent_space)

            pickle.dump(tsne_features, open(tsne_features_path, 'wb'))
        else:
            print('Nothing found ...')

    if tsne_features.any():
        plt.figure(figsize=(8, 6), dpi=100)
        plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=labels, edgecolors='none')
        plt.title(os.path.splitext(tsne_features_path)[0])
        plt.colorbar()
        plt.show()
else:
    print('No labels')
