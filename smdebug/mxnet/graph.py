# Standard Library
import json

# Third Party
from mxnet.gluon import HybridBlock
from mxnet.symbol import Symbol

# First Party
from smdebug.core.tfevent.proto.attr_value_pb2 import AttrValue
from smdebug.core.tfevent.proto.graph_pb2 import GraphDef
from smdebug.core.tfevent.proto.node_def_pb2 import NodeDef
from smdebug.core.tfevent.proto.versions_pb2 import VersionDef


def _scoped_name(scope_name, node_name):
    return "/".join([scope_name, node_name])


def _get_nodes_from_symbol(sym):
    """Given a symbol and shapes, return a list of `NodeDef`s for visualizing the
    the graph in TensorBoard."""
    if not isinstance(sym, Symbol):
        raise TypeError(
            "sym must be an `mxnet.symbol.Symbol`," " received type {}".format(str(type(sym)))
        )
    conf = json.loads(sym.tojson())
    nodes = conf["nodes"]
    data2op = {}  # key: data id, value: list of ops to whom data is an input
    for i, node in enumerate(nodes):
        if node["op"] != "null":  # node is an operator
            input_list = node["inputs"]
            for idx in input_list:
                if idx[0] == 0:  # do not include 'data' node in the op scope
                    continue
                if idx[0] in data2op:
                    # nodes[idx[0]] is a data as an input to op nodes[i]
                    data2op[idx[0]].append(i)
                else:
                    data2op[idx[0]] = [i]

    # In the following, we group data with operators they belong to
    # by attaching them with operator names as scope names.
    # The parameters with the operator name as the prefix will be
    # assigned with the scope name of that operator. For example,
    # a convolution op has name 'conv', while its weight and bias
    # have name 'conv_weight' and 'conv_bias'. In the end, the operator
    # has scope name 'conv' prepended to its name, i.e. 'conv/conv'.
    # The parameters are named 'conv/conv_weight' and 'conv/conv_bias'.
    node_defs = []
    for i, node in enumerate(nodes):
        node_name = node["name"]
        op_name = node["op"]
        kwargs = {"op": op_name, "name": node_name}
        if op_name != "null":  # node is an operator
            inputs = []
            input_list = node["inputs"]
            for idx in input_list:
                input_node = nodes[idx[0]]
                input_node_name = input_node["name"]
                if input_node["op"] != "null":
                    inputs.append(_scoped_name(input_node_name, input_node_name))
                elif idx[0] in data2op and len(data2op[idx[0]]) == 1 and data2op[idx[0]][0] == i:
                    # the data is only as an input to nodes[i], no else
                    inputs.append(_scoped_name(node_name, input_node_name))
                else:  # the data node has no scope name, e.g. 'data' as the input node
                    inputs.append(input_node_name)
            kwargs["input"] = inputs
            kwargs["name"] = _scoped_name(node_name, node_name)
        elif i in data2op and len(data2op[i]) == 1:
            # node is a data node belonging to one op, find out which operator this node belongs to
            op_node_name = nodes[data2op[i][0]]["name"]
            kwargs["name"] = _scoped_name(op_node_name, node_name)

        if "attrs" in node:
            # TensorBoard would escape quotation marks, replace it with space
            attr = json.dumps(node["attrs"], sort_keys=True).replace('"', " ")
            attr = {"param": AttrValue(s=attr.encode(encoding="utf-8"))}
            kwargs["attr"] = attr
        node_def = NodeDef(**kwargs)
        node_defs.append(node_def)
    return node_defs


def _sym2pb(sym):
    """Converts an MXNet symbol to its graph protobuf definition."""
    return GraphDef(node=_get_nodes_from_symbol(sym), versions=VersionDef(producer=100))


def _net2pb(net):
    if isinstance(net, HybridBlock):
        # TODO(junwu): may need a more approprite way to get symbol from a HybridBlock
        if not net._cached_graph:
            raise RuntimeError(
                "Please first call net.hybridize() and then run forward with "
                "this net at least once before calling add_graph()."
            )
        net = net._cached_graph[1]
    elif not isinstance(net, Symbol):
        raise TypeError(
            "only accepts mxnet.gluon.HybridBlock and mxnet.symbol.Symbol "
            "as input network, received type {}".format(str(type(net)))
        )
    return _sym2pb(net)
