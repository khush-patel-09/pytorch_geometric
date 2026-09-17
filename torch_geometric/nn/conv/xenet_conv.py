import torch

from collections.abc import Sequence
from typing import Optional, Union

from torch import Tensor
from torch import nn

from torch_geometric.nn.conv import MessagePassing


class XENetConv(MessagePassing):
    r"""XENet graph convolution.

    XENet uses node features and edge features to compute edge-level
    representations, which are then aggregated in incoming and outgoing
    directions to update node features.

    Args:
        stack_channels (int or list[int]): Hidden channels of the
            edge-message stack.
        node_channels (int): Number of output node channels.
        edge_channels (int): Number of output edge channels.
        attention (bool, optional): Whether to use attention during
            message aggregation. (default: ``True``)
        bias (bool, optional): Whether to use bias terms.
            (default: ``True``)
    """

    def __init__(
        self,
        stack_channels: Union[int, Sequence[int]],
        node_channels: int,
        edge_channels: int,
        attention: bool = True,
        bias: bool = True,
    ):
        super().__init__(aggr="sum")

        if isinstance(stack_channels, int):
            stack_channels = [stack_channels]

        self.stack_channels = list(stack_channels)
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.attention = attention

        self.stack_models = nn.ModuleList()
        self.node_model = None
        self.edge_model = None

        if attention:
            self.incoming_attention = None
            self.outgoing_attention = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Runs the forward pass.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Graph connectivity.
            edge_attr (torch.Tensor): Edge features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                Updated node features and edge features.
        """
        raise NotImplementedError

    def message(self, stack: Tensor) -> Tensor:
        """Constructs messages for node aggregation."""
        raise NotImplementedError

    @staticmethod
    def _get_reverse_edge_index(
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        """Returns the index of the reverse edge for every edge."""

        src, dst = edge_index

        # Encode each directed edge (src, dst) as a unique integer.
        keys = src * num_nodes + dst
        reverse_keys = dst * num_nodes + src

        # Sort edge keys so we can efficiently locate reverse edges.
        sorted_keys, permutation = torch.sort(keys)

        positions = torch.searchsorted(
            sorted_keys,
            reverse_keys,
        )

        valid = positions < sorted_keys.numel()

        if valid.any():
            valid_positions = positions[valid]
            matches = (
                sorted_keys[valid_positions] == reverse_keys[valid]
            )
            valid_indices = valid.nonzero(as_tuple=True)[0]
            valid[valid_indices] = matches

        if not valid.all():
            raise ValueError(
                "XENetConv requires both directions of every edge. "
                "For every edge i -> j, the reverse edge j -> i "
                "must also be present in edge_index."
            )

        return permutation[positions]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"stack_channels={self.stack_channels}, "
            f"node_channels={self.node_channels}, "
            f"edge_channels={self.edge_channels})"
        )