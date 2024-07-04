import torch
import torch.nn.functional as F


def aggregate_features(a, b, w, k):
    """
    a: Tensor of shape (batch_size, feature_dim)   #  text feature
    b: Tensor of shape (batch_size, num_features, feature_dim)
    w: Tensor of shape (batch_size, num_features)    #  Original attention

    Returns: Aggregated feature tensor of shape (batch_size, feature_dim)
    """

    # 1. Compute cosine similarity
    similarity = F.cosine_similarity(a.unsqueeze(1), b, dim=2)  # Shape: (batch_size, num_features)

    # 2. Find top 5
    _, top_indices = torch.topk(similarity, k, dim=1)  # Shape: (batch_size, 5)

    # 3. Use the weight w to aggregate the top 5 features for each batch
    batch_size = a.size(0)
    aggregated_features_list = []

    for i in range(batch_size):
        top_features_for_batch = b[i, top_indices[i]]  # Shape: (5, feature_dim)
        weights_for_batch = w[i, top_indices[i]].unsqueeze(-1)  # Shape: (5, 1)

        weighted_features = top_features_for_batch * weights_for_batch  # Shape: (5, feature_dim)
        aggregated_feature = torch.sum(weighted_features, dim=0, keepdim=True)  # Shape: (1, feature_dim)

        aggregated_features_list.append(aggregated_feature)

    aggregated_features = torch.cat(aggregated_features_list, dim=0)  # Shape: (batch_size, feature_dim)

    return aggregated_features


# Example tensors
a = torch.rand(4, 32)  # Batch size of 4
b = torch.rand(4, 5, 32)  # Batch size of 4
w = torch.rand(4, 5)  # Batch size of 4

result = aggregate_features(a, b, w, 2)
print(result.shape)
