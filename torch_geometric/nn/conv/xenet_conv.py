import torch

from collections.abc import Sequence
from typing import Optional, Union

from torch import Tensor
from torch import nn

from torch_geometric.nn import Linear, MessagePassing


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

        for i, channels in enumerate(self.stack_channels):
            in_channels = -1 if i == 0 else self.stack_channels[i - 1]
            self.stack_models.append(
                Linear(in_channels, channels)
            )

        self.stack_activation = nn.PReLU(self.stack_channels[-1])

        self.node_model = None
        self.edge_model = None

        if attention:
            self.incoming_attention = Linear(
                self.stack_channels[-1],
                1,
            )
            self.outgoing_attention = Linear(
                self.stack_channels[-1],
                1,
            )


    def _compute_stack(
        self,
        x_i: Tensor,
        x_j: Tensor,
        e_ij: Tensor,
        e_ji: Tensor,
    ) -> Tensor:
        """Computes the XENet edge representation s_ij."""

        stack = torch.cat(
            [x_i, x_j, e_ij, e_ji],
            dim=-1,
        )

        for model in self.stack_models:
            stack = model(stack)
            stack = self.stack_activation(stack)

        return stack
    

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Runs the forward pass."""

        if x.dim() != 2:
            raise ValueError(
                "'x' must have shape [num_nodes, num_node_features]"
            )

        if edge_index.dim() != 2 or edge_index.size(0) != 2:
            raise ValueError(
                "'edge_index' must have shape [2, num_edges]"
            )

        if edge_attr.dim() == 1:
            edge_attr = edge_attr.view(-1, 1)

        if edge_attr.dim() != 2:
            raise ValueError(
                "'edge_attr' must have shape [num_edges, num_edge_features]"
            )

        if edge_index.size(1) != edge_attr.size(0):
            raise ValueError(
                "'edge_index' and 'edge_attr' must contain the same "
                "number of edges"
            )

        reverse_index = self._get_reverse_edge_index(
            edge_index,
            x.size(0),
        )

        src, dst = edge_index

        x_i = x[src]
        x_j = x[dst]

        e_ij = edge_attr
        e_ji = edge_attr[reverse_index]

        stack = self._compute_stack(
            x_i,
            x_j,
            e_ij,
            e_ji,
        )

        # Incoming: j -> i
        incoming = self.propagate(
            edge_index,
            stack=stack,
            direction="incoming",
        )

        # Outgoing: i -> j
        outgoing = self.propagate(
            edge_index.flip(0),
            stack=stack,
            direction="outgoing",
        )

        # Node and edge updates will be implemented in the next commit.
        raise NotImplementedError(
            "Node and edge updates are not implemented yet."
        )

    def message(
        self,
        stack: Tensor,
        direction: str,
    ) -> Tensor:
        """Constructs incoming or outgoing messages."""

        if not self.attention:
            return stack

        if direction == "incoming":
            attention = torch.sigmoid(
                self.incoming_attention(stack)
            )
        elif direction == "outgoing":
            attention = torch.sigmoid(
                self.outgoing_attention(stack)
            )
        else:
            raise ValueError(f"Unknown direction: {direction}")

        return stack * attention

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