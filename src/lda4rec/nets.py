# Todo: Rename to nets
"""
Different model architectures

Note: The BilinearNet class is more or less taken from Spotlight (MIT)
"""
import torch
import torch.nn as nn


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.normal_(0, 1.0 / self.embedding_dim)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class ZeroEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a normal variable scaled by the inverse
    of the embedding dimension.

    Used for biases.
    """

    def reset_parameters(self):
        """
        Initialize parameters.
        """

        self.weight.data.zero_()
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class MFNet(nn.Module):
    """
    Bilinear factorization representation.
    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    n_users (int): Number of users in the model.
    n_items (int): Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    biases (bool):
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(
        self,
        n_users,
        n_items,
        *,
        embedding_dim=32,
        user_embedding_layer=None,
        item_embedding_layer=None,
        sparse=False
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(
                n_users, embedding_dim, sparse=sparse
            )

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(
                n_items, embedding_dim, sparse=sparse
            )

        self.user_biases = ZeroEmbedding(n_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(n_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        dot = user_embedding * item_embedding
        if dot.dim() > 1:  # handles case where embedding_dim=1
            dot = dot.sum(1)

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()
        dot = dot + user_bias + item_bias

        return dot


class SNMFNet(nn.Module):
    """
    Semi-Positive Bilinear factorization representation.
    Encodes both users and items as an embedding layer; the score
    for a user-item pair is given by the dot product of the item
    and user latent vectors.

    n_users (int): Number of users in the model.
    n_items (int): Number of items in the model.
    embedding_dim: int, optional
        Dimensionality of the latent representations.
    biases (bool):
    user_embedding_layer: an embedding layer, optional
        If supplied, will be used as the user embedding layer
        of the network.
    item_embedding_layer: an embedding layer, optional
        If supplied, will be used as the item embedding layer
        of the network.
    sparse: boolean, optional
        Use sparse gradients.
    """

    def __init__(
        self,
        n_users,
        n_items,
        *,
        embedding_dim=32,
        user_embedding_layer=None,
        item_embedding_layer=None,
        sparse=False
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        if user_embedding_layer is not None:
            self.user_embeddings = user_embedding_layer
        else:
            self.user_embeddings = ScaledEmbedding(
                n_users, embedding_dim, sparse=sparse
            )

        if item_embedding_layer is not None:
            self.item_embeddings = item_embedding_layer
        else:
            self.item_embeddings = ScaledEmbedding(
                n_items, embedding_dim, sparse=sparse
            )

        self.user_biases = ZeroEmbedding(n_users, 1, sparse=sparse)
        self.item_biases = ZeroEmbedding(n_items, 1, sparse=sparse)

    def forward(self, user_ids, item_ids):
        """
        Compute the forward pass of the representation.

        Parameters
        ----------

        user_ids: tensor
            Tensor of user indices.
        item_ids: tensor
            Tensor of item indices.

        Returns
        -------

        predictions: tensor
            Tensor of predictions.
        """

        user_embedding = self.user_embeddings(user_ids)
        item_embedding = self.item_embeddings(item_ids)

        user_embedding = user_embedding.squeeze()
        item_embedding = item_embedding.squeeze()

        user_bias = self.user_biases(user_ids).squeeze()
        item_bias = self.item_biases(item_ids).squeeze()

        dot = user_embedding * torch.sigmoid(item_embedding)

        if dot.dim() > 1:  # handles case where embedding_dim=1
            dot = dot.sum(1)

        dot = dot + user_bias + item_bias
        return dot
