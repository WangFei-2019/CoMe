import os
import time
import json
import pickle
from tqdm import tqdm

import torch
from sklearn.feature_selection import mutual_info_regression
from sklearn.neighbors import NearestNeighbors

from utils.data_utils import get_trainloaders
from utils.eval_utils import load_and_eval_ppl, eval_zero_shot
from utils.util import *

def adaptive_chunk_size(total_size, preferred_size=100):
    """
    Determines the optimal chunk size for processing to maximize efficiency.

    Args:
        total_size (int): The total number of elements to process.
        preferred_size (int, optional): The preferred chunk size. Defaults to 100.

    Returns:
        int: The adaptive chunk size.
    """
    # Iterate from preferred_size down to 1 to find the largest divisor of total_size
    for size in range(preferred_size, 0, -1):
        if total_size % size == 0:
            return size
    return 1  # Fallback to 1 if no divisor is found

def L2_distance_chunked(a, b, df, total_size):
    """
    Generates L2 distance chunks between two arrays in an adaptive chunked manner.

    Args:
        a (np.ndarray): First array of shape (n_samples_a, n_features).
        b (np.ndarray): Second array of shape (n_samples_b, n_features).
        df (int): Flag to determine if diagonal should be zeroed.
        total_size (int): Total number of samples.

    Yields:
        np.ndarray: A chunk of L2 distances.
    """
    # Determine the chunk size adaptively
    chunk_size = adaptive_chunk_size(total_size)
    # Reshape a and b if they have more than 2 dimensions
    if a.ndim > 2:
        a = a.reshape(-1, a.shape[-1])
    if b.ndim > 2:
        b = b.reshape(-1, b.shape[-1])

    # Ensure a and b have the same number of features
    assert a.shape[1] == b.shape[1], "Incompatible shapes"

    # Iterate over chunks of a
    for i in range(0, a.shape[0], chunk_size):
        # Compute squared norms for the current chunk of a
        aa = np.sum(a[i : i + chunk_size] ** 2, axis=1, keepdims=True)
        # Iterate over chunks of b
        for j in range(0, b.shape[0], chunk_size):
            # Compute squared norms for the current chunk of b
            bb = np.sum(b[j : j + chunk_size] ** 2, axis=1, keepdims=True).T
            # Compute the dot product between chunks of a and b
            ab = a[i : i + chunk_size] @ b[j : j + chunk_size].T
            # Compute the L2 distance chunk
            d_chunk = np.sqrt(np.abs(aa + bb - 2 * ab))

            # If df flag is set to 1 and processing diagonal chunks, set diagonal to 0
            if df == 1:
                if i == j:
                    np.fill_diagonal(d_chunk, 0)  # Set diagonal to 0 if needed

            # Yield the computed distance chunk
            yield d_chunk


def diffusionKernel(X, sigmaK, alpha, d, total_size):
    """
    Computes the diffusion kernel embedding for the dataset X.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).
        sigmaK (float): Kernel scale parameter.
        alpha (float): Scaling factor for normalization.
        d (int): Target dimensionality for embedding.
        total_size (int): Total number of samples.

    Returns:
        np.ndarray: Embedded data of shape (n_samples, d).
    """
    # Determine the optimal chunk size for processing
    chunk_size = adaptive_chunk_size(total_size)
    logging.info("Starting diffusion kernel computation...")
    kernel_start_time = time.time()

    n = X.shape[0]  # Number of samples
    # Initialize the kernel matrix with zeros
    K = np.zeros((n, n), dtype=np.float32)

    # Iterate over chunks of X to compute the kernel matrix
    for i in range(0, n, chunk_size):
        for j in range(0, n, chunk_size):
            i_end = min(i + chunk_size, n)
            j_end = min(j + chunk_size, n)
            # Compute the L2 distance chunk between X[i:i_end] and X[j:j_end]
            D_chunk = next(L2_distance_chunked(X[i:i_end], X[j:j_end], df=1, total_size=n))
            # Compute the kernel chunk using the diffusion kernel formula
            K_chunk = np.exp(-((D_chunk / sigmaK) ** 0.5))
            # Assign the computed chunk to the appropriate position in K
            K[i:i_end, j:j_end] = K_chunk[: i_end - i, : j_end - j]

    # Calculate the sum of the kernel matrix along columns
    p = np.sum(K, axis=0)
    # Normalize the kernel matrix
    K1 = K / (p * p.reshape(-1, 1)) ** alpha
    # Compute the normalization factor
    v = np.sqrt(np.sum(K1, axis=0))
    # Normalize the kernel matrix further
    A = K1 / np.outer(v, v)

    # Compute the condition number of the matrix A for numerical stability
    cond_num = np.linalg.cond(A)
    logging.info(f"Condition number: {cond_num}")

    # If the condition number is infinite, apply regularization to stabilize
    if np.isinf(cond_num):
        logging.info("Infinite condition number detected. Applying regularization...")
        regularization = 1e-6
        max_iterations = 10
        iteration = 0
        while np.isinf(cond_num) and iteration < max_iterations:
            # Add a small value to the diagonal for regularization
            A += np.eye(A.shape[0]) * regularization
            cond_num = np.linalg.cond(A)
            regularization *= 10  # Increase regularization factor exponentially
            iteration += 1
        logging.info(f"Regularization applied. New condition number: {cond_num}")

    # Replace any NaNs in A with zero
    A = np.nan_to_num(A)

    # Handle very small values by setting them to a minimum threshold
    zero_mask = np.abs(A) < 1e-12
    A[zero_mask] = 1e-12

    # Perform Singular Value Decomposition (SVD) on the matrix A
    U, S, V = np.linalg.svd(A, full_matrices=False)
    # Retain only the top (d + 1) singular vectors
    U = U[:, :d + 1]
    # Avoid division by zero by replacing zeros in the first column
    U[:, 0] = np.where(U[:, 0] == 0, 1e-8, U[:, 0])
    # Normalize U by the first column
    U = U / U[:, 0].reshape(-1, 1)

    # Extract the embedded coordinates excluding the first column
    Y = U[:, 1 : d + 1]

    kernel_end_time = time.time()
    logging.info(f"Diffusion kernel computation completed in {kernel_end_time - kernel_start_time:.2f} seconds.")
    return Y

def entropy_estimator_knn(x, k=1):
    """
    Estimates the entropy of the dataset x using a k-nearest neighbors approach.

    Args:
        x (np.ndarray): Input data of shape (n_samples, n_features).
        k (int, optional): Number of neighbors to consider. Defaults to 1.

    Returns:
        float: Estimated entropy.
    """
    n, d = x.shape  # Number of samples and dimensions
    # Initialize the NearestNeighbors model
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(x)
    # Compute the distances to the nearest neighbors
    distances, _ = nbrs.kneighbors(x)
    # Take the distance to the k-th neighbor (excluding the point itself)
    distances = distances[:, -1]
    # Compute the entropy estimate using the KNN formula
    return -np.mean(np.log(k / (n * distances**d)))

def compute_similarity_matrix_npib_global(embeddings, n_neighbors=5, k_entropy=50):
    """
    Computes a similarity matrix between different layers based on normalized pointwise information bottleneck (NPIB).

    Args:
        embeddings (list): List of NumPy arrays containing embeddings for each layer.
        n_neighbors (int, optional): Number of neighbors for mutual information computation. Defaults to 5.
        k_entropy (int, optional): Number of neighbors for entropy estimation. Defaults to 50.

    Returns:
        np.ndarray: The computed similarity matrix of shape (num_layers, num_layers).
    """
    num_layers = len(embeddings)  # Number of layers
    # Initialize the similarity matrix with zeros
    similarity_matrix = np.zeros((num_layers, num_layers))

    # Iterate over each pair of layers
    for i in tqdm(range(num_layers), total=num_layers, desc="Compute Similarity Matrix NPIB:"):
        for j in range(i, num_layers):
            emb_i = embeddings[i]  # Embeddings for layer i
            emb_j = embeddings[j]  # Embeddings for layer j

            # Ensure both embeddings have the same number of samples by taking the minimum
            min_samples = min(emb_i.shape[0], emb_j.shape[0])
            emb_i = emb_i[:min_samples, :]
            emb_j = emb_j[:min_samples, :]

            # List to store mutual information scores for each dimension
            mi_scores = []
            # Compute mutual information between each dimension of emb_j and the entire emb_i
            for dim in range(emb_j.shape[1]):
                mi_score = mutual_info_regression(
                    emb_i,
                    emb_j[:, dim],
                    discrete_features=False,
                    n_neighbors=n_neighbors,
                )
                # Take the mean mutual information score for the current dimension
                mi_scores.append(np.mean(mi_score))

            # Compute the average mutual information across all dimensions
            mutual_info = np.mean(mi_scores)
            # Estimate the entropy for both embeddings
            entropy_i = entropy_estimator_knn(emb_i, k=k_entropy)
            entropy_j = entropy_estimator_knn(emb_j, k=k_entropy)
            # Compute the normalized pointwise information bottleneck (NPIB)
            npib = mutual_info / np.sqrt(entropy_i * entropy_j)

            # Assign the computed similarity to the matrix (symmetrically)
            similarity_matrix[i, j] = npib
            similarity_matrix[j, i] = npib

    return similarity_matrix

def compute_fusion_ratios(similarity_matrix, sorted_pairs, beta=1.0):
    """
    Computes fusion ratios based on the similarity matrix and sorted layer pairs.

    Args:
        similarity_matrix (np.ndarray): The similarity matrix between layers.
        sorted_pairs (list of tuples): List of layer index pairs to fuse.
        beta (float, optional): Scaling factor for the fusion ratio. Defaults to 1.0.

    Returns:
        list of tuples: List containing (ratio_i, ratio_j) for each pair.
    """
    fusion_ratios = []  # List to store fusion ratios for each pair
    similaritys = []
    # Iterate over each sorted pair of layers
    for i, j in sorted_pairs:
        # Compute the mean similarity for each layer across all other layers
        similarity_i = np.mean(similarity_matrix[i, :])
        similarity_j = np.mean(similarity_matrix[j, :])
        # Compute the total similarity for normalization
        total_similarity = similarity_i + similarity_j

        # Calculate the ratio for each layer based on their similarity
        ratio_i = similarity_i / total_similarity
        ratio_j = similarity_j / total_similarity

        # Apply a sigmoid-like adjustment to the ratios using beta
        adjusted_ratio_i = np.exp(beta * ratio_i) / (1 + np.exp(beta * ratio_i))
        adjusted_ratio_j = 1 - adjusted_ratio_i

        # Append the adjusted ratios as a tuple
        fusion_ratios.append((adjusted_ratio_i, adjusted_ratio_j))
        similaritys.append((similarity_i, similarity_j))

    return fusion_ratios, similaritys

@torch.inference_mode()
def main_func(args, modelhander):
    if args.nsamples % args.num_tasks!=0:
        raise ValueError(f"The number of samples for each category must be the same, and 'nsamples'({args.nsamples}) must be divisible by 'num_tasks'({args.num_tasks}).")
    dataloader = get_trainloaders(args.calibration_dataset,
                                  tokenizer=modelhander.tokenizer,
                                  nsamples=args.nsamples,
                                  seed=args.seed,
                                  seqlen=modelhander.model.seqlen,
                                  num_tasks=args.num_tasks
                                  )

    batch_size = 1
    # Get input IDs
    testenc = dataloader.input_ids
    testattn = dataloader.attention_mask

    # Calculate number of samples
    nsamples = testenc.numel() // modelhander.model.seqlen
    layer_num = modelhander.model.config.num_hidden_layers
    activations = [[] for i in range(layer_num)]
    start_time = time.time()
    # Loop through each batch
    for i in tqdm(range(0, nsamples, batch_size), desc="Get Hidden states ..."):
        # Calculate end index
        j = min(i+batch_size, nsamples)
        # Prepare inputs and move to device
        inputs = testenc[:, (i * modelhander.model.seqlen):(j * modelhander.model.seqlen)].to(modelhander.model.device)
        inputs = inputs.reshape(j-i, modelhander.model.seqlen)

        attn = testattn[:, (i * modelhander.model.seqlen):(j * modelhander.model.seqlen)].to(modelhander.model.device)
        attn = attn.reshape(j-i, modelhander.model.seqlen)

        # Forward pass through the model
        hidden_states = modelhander.model(inputs, attention_mask=attn, output_hidden_states=True).hidden_states[1:]
        for idx in range(layer_num):
            activations[idx].append(hidden_states[idx][:, :512, :].to(torch.float32).cpu())

    activations = [np.concatenate(i, axis=0) for i in activations]

    hidden_states_end_time = time.time()

    embeddings = [] 
    for layer_idx in tqdm(range(layer_num), desc="DiffusionKernel process"):
        embedded_activations = diffusionKernel(activations[layer_idx], sigmaK=8, alpha=0.5, d=2, total_size=activations[layer_idx].shape[0])

        # Define the output file path for the embedded activations
        output_file = os.path.join(args.save_path, f"layer_{layer_idx}_embedded.pkl")
        # Save the embedded activations to a pickle file
        with open(output_file, "wb") as f:
            pickle.dump(embedded_activations, f)

        embedded_activations = np.nan_to_num(embedded_activations, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply rank normalization to the embeddings
        embedded_activations = (
            np.argsort(np.argsort(embedded_activations, axis=0), axis=0)
            / embedded_activations.shape[0]
        )
        embeddings.append(embedded_activations)

    # Compute the similarity matrix based on the loaded embeddings
    similarity_matrix = compute_similarity_matrix_npib_global(embeddings, n_neighbors=args.nsamples//args.num_tasks, k_entropy=args.num_tasks)
    end_time = time.time()

    logging.info("#"*20, " Start Layer Merge ", "#"*20)
    mka_info = {"total_time": end_time-start_time, "hidden_states_time": hidden_states_end_time-start_time, "diffusionKernel&similarity_calculated_time": end_time-hidden_states_end_time}

    while modelhander.model.config.num_hidden_layers > args.target_layers:
        logging.info("*"*10, f"From {modelhander.model.config.num_hidden_layers} layers to {modelhander.model.config.num_hidden_layers-1}", "*"*10)
        merge_layer1_idx = modelhander.model.config.num_hidden_layers - 2
        merge_layer2_idx = modelhander.model.config.num_hidden_layers - 1
        fusion_ratios, similaritys = compute_fusion_ratios(similarity_matrix, [(merge_layer1_idx, merge_layer2_idx)])

        state_dict = modelhander.add_heads([merge_layer1_idx, merge_layer2_idx], fusion_ratios[0])
        state_dict.update(modelhander.add_neuron([merge_layer1_idx, merge_layer2_idx], fusion_ratios[0]))

        modelhander.adjust_layer_index(merge_index_list=[merge_layer1_idx, merge_layer2_idx], state_dict=state_dict)

        mka_info[f"prune_to_{modelhander.model.config.num_hidden_layers}"] = {}

        logging.info(f"Merging Layer {merge_layer1_idx} (simi {similaritys[0][0]:.6f}) and Layer {merge_layer2_idx} (simi {similaritys[0][1]:.6f}). \nFusion Ratio: {fusion_ratios[0][0]:.4f} : {fusion_ratios[0][1]:.4f}")

        mka_info[f"prune_to_{modelhander.model.config.num_hidden_layers}"]["merge_list"] = [merge_layer1_idx, merge_layer2_idx]
        mka_info[f"prune_to_{modelhander.model.config.num_hidden_layers}"]["merge_ratio"] = fusion_ratios[0]
        mka_info[f"prune_to_{modelhander.model.config.num_hidden_layers}"]["similaritys"] = similaritys[0]

        if args.continue_saving:
            save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}")
            modelhander.save(path=save_path)
            logging.info(f"Save model to {save_path}")

            for da in args.ppl_data:
                logging.info(f"Starting {da} PPL evaluation...")
                ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
                mka_info[f"prune_to_{modelhander.model.config.num_hidden_layers}"][f"{da}_ppl"] = ppl

    if not args.continue_saving:
        save_path = os.path.join(args.save_path, args.save_name + f"_{modelhander.model.config.num_hidden_layers}")
        modelhander.save(path=save_path)
        logging.info(f"Save model to {save_path}")

        for da in args.ppl_data:
            logging.info(f"Starting {da} PPL evaluation...")
            ppl = load_and_eval_ppl(modelhander.model, dataset=da, tokenizer=modelhander.tokenizer)
            mka_info[f"prune_to_{modelhander.model.config.num_hidden_layers}"][f"{da}_ppl"] = ppl
    
    logging.info(f"Save {args.method} information to {os.path.join(args.save_path, f'{args.method}_info.json')}")
    with open(os.path.join(args.save_path, f'{args.method}_info.json'), 'w') as json_file:
        json.dump(mka_info, json_file, indent=4)

