import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment

from clusgan.definitions import DATASETS_DIR, RUNS_DIR
from clusgan.models import Encoder_CNN
from clusgan.datasets import get_dataloader


def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.size == y_pred.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() / y_pred.size


def evaluate(y_true, y_pred, name="method"):
    acc = cluster_acc(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)

    print(f"{name}")
    print(f"  ACC: {acc:.4f}")
    print(f"  NMI: {nmi:.4f}")
    print(f"  ARI: {ari:.4f}")
    print("-" * 40)

    return {"method": name, "ACC": acc, "NMI": nmi, "ARI": ari}


def main():
    parser = argparse.ArgumentParser(description="Evaluate clustering metrics on MNIST")
    parser.add_argument("-r", "--run_dir", required=True, help="Run directory, e.g. mnist\\100epoch_z30_van_bs64_test_run100")
    parser.add_argument("-n", "--n_samples", type=int, default=10000, help="Number of test samples to use")
    parser.add_argument("--pca_dim", type=int, default=50, help="PCA dim for raw-pixel baseline")
    args = parser.parse_args()

    # Device
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda else "cpu")
    print(f"Using device: {device}")

    # Parse run path
    run_dir_arg = os.path.normpath(args.run_dir)
    run_name = os.path.basename(run_dir_arg)
    dataset_name = os.path.basename(os.path.dirname(run_dir_arg))

    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    models_dir = os.path.join(run_dir, "models")

    # Read training metadata
    train_df = pd.read_csv(os.path.join(run_dir, "training_details.csv"))
    latent_dim = int(train_df["latent_dim"][0])
    n_c = int(train_df["n_classes"][0])

    # Load encoder
    encoder = Encoder_CNN(latent_dim, n_c)
    enc_path = os.path.join(models_dir, encoder.name + ".pth.tar")
    encoder.load_state_dict(torch.load(enc_path, map_location=device))
    encoder = encoder.to(device)
    encoder.eval()

    # Load test data
    dataloader = get_dataloader(
        dataset_name=dataset_name,
        data_dir=data_dir,
        batch_size=args.n_samples,
        train_set=False
    )

    imgs, labels = next(iter(dataloader))
    imgs = Variable(imgs, requires_grad=False).to(device)
    y_true = labels.cpu().numpy()

    # Flatten raw pixels for raw baselines
    X_raw = imgs.detach().cpu().numpy().reshape(imgs.size(0), -1)

    # Encode latent representations
    with torch.no_grad():
        enc_zn, enc_zc, enc_zc_logits = encoder(imgs)

    zn = enc_zn.detach().cpu().numpy()
    zc_logits = enc_zc_logits.detach().cpu().numpy()
    latent_full = np.hstack([zn, zc_logits])

    results = []

    # -------------------------------------------------
    # 1) ClusterGAN direct prediction: argmax on zc logits
    # -------------------------------------------------
    y_pred_clustergan = np.argmax(zc_logits, axis=1)
    results.append(evaluate(y_true, y_pred_clustergan, "ClusterGAN_direct_argmax"))

    # -------------------------------------------------
    # 2) KMeans on latent
    # -------------------------------------------------
    km_latent = KMeans(n_clusters=n_c, random_state=0, n_init=20)
    y_pred_km_latent = km_latent.fit_predict(latent_full)
    results.append(evaluate(y_true, y_pred_km_latent, "KMeans_on_latent"))

    # -------------------------------------------------
    # 3) GMM on latent
    # -------------------------------------------------
    gmm_latent = GaussianMixture(n_components=n_c, random_state=0)
    y_pred_gmm_latent = gmm_latent.fit_predict(latent_full)
    results.append(evaluate(y_true, y_pred_gmm_latent, "GMM_on_latent"))

    # -------------------------------------------------
    # 4) Agglomerative on latent
    # -------------------------------------------------
    agg_latent = AgglomerativeClustering(n_clusters=n_c)
    y_pred_agg_latent = agg_latent.fit_predict(latent_full)
    results.append(evaluate(y_true, y_pred_agg_latent, "Agglomerative_on_latent"))

    # -------------------------------------------------
    # 5) Spectral on latent
    # -------------------------------------------------
    spec_latent = SpectralClustering(
        n_clusters=n_c,
        affinity="nearest_neighbors",
        random_state=0,
        assign_labels="kmeans"
    )
    y_pred_spec_latent = spec_latent.fit_predict(latent_full)
    results.append(evaluate(y_true, y_pred_spec_latent, "Spectral_on_latent"))

    # -------------------------------------------------
    # 6) KMeans on raw pixels after PCA
    # -------------------------------------------------
    pca = PCA(n_components=args.pca_dim, random_state=0)
    X_pca = pca.fit_transform(X_raw)

    km_raw = KMeans(n_clusters=n_c, random_state=0, n_init=20)
    y_pred_km_raw = km_raw.fit_predict(X_pca)
    results.append(evaluate(y_true, y_pred_km_raw, f"KMeans_on_raw_PCA{args.pca_dim}"))

    # Save results
    results_df = pd.DataFrame(results)
    out_csv = os.path.join(run_dir, "clustering_eval_results.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")


if __name__ == "__main__":
    main()