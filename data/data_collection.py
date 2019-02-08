import json
import os
import sys
import codecs

from data.graph_pb2 import Graph
from data.graph_pb2 import FeatureNode


def stringify_node(node):
    node_dict = {
        'contents': node.contents,
        'startPosition': node.startPosition,
        'endPosition': node.endPosition,
        'id': node.id,
        'type': node.type,
        'startLineNumber': node.startLineNumber,
        'endLineNumber': node.endLineNumber
    }

    return node_dict


def graph_to_map(graph):
    id_to_node = {}
    id_to_connections = {}

    for node in graph.node:
        id_to_node[node.id] = node
        id_to_connections[node.id] = set()

    for edge in graph.edge:
        id_to_connections[edge.sourceId] = id_to_connections[edge.sourceId].union({edge.destinationId})

    for key in id_to_connections:
        id_to_connections[key] = list(id_to_connections[key])

    return id_to_node, id_to_connections


def get_tokens(nodes, start_position, end_position):
    tokens = []
    for node in nodes:
        if node.type == FeatureNode.IDENTIFIER_TOKEN or node.type == FeatureNode.TOKEN:
            if node.startPosition >= start_position and node.endPosition <= end_position:
                tokens.append(node.contents)
    return tokens


def clean_javadoc(javadoc_str):
    lines = javadoc_str.split("\n")
    lines = [line for line in lines if "@param" not in line]
    stringed_lines = '\n'.join(lines)

    expression_to_replace = ["@link", "@code", "@inheritDoc"]
    for expr in expression_to_replace:
        stringed_lines = stringed_lines.replace(expr, "")
    return stringed_lines


# [{
#     'type': variable_type,
#     'name': variable_name
# }]
def get_fields(id_to_node, id_to_connections):
    fields = []

    for node in id_to_node.values():
        if node.type == FeatureNode.FAKE_AST and node.contents == "MEMBERS":
            for connection_id in id_to_connections[node.id]:
                if id_to_node[connection_id].contents == "VARIABLE":
                    field = {}
                    for field_child in id_to_connections[connection_id]:
                        field_child_node = id_to_node[field_child]
                        if field_child_node.contents == "TYPE":
                            field['type'] = get_tokens(
                                id_to_node.values(),
                                field_child_node.startPosition,
                                field_child_node.endPosition
                            )[0]

                        if field_child_node.type == FeatureNode.IDENTIFIER_TOKEN:
                            field['name'] = field_child_node.contents
                    if 'type' in field and 'name' in field:
                        fields.append(field)
    return fields


def get_subtree(root_id, id_to_node, id_to_connections):
    import copy

    nodes_in_subtree = get_subtree_dfs(root_id, id_to_node, id_to_connections)

    new_id_to_node = copy.deepcopy(id_to_node)
    new_id_to_connections = copy.deepcopy(id_to_connections)

    for node_id in id_to_node:
        if node_id not in nodes_in_subtree:
            new_id_to_node.pop(node_id)
            new_id_to_connections.pop(node_id)

    for key in new_id_to_connections:
        new_id_to_connections[key] = [x for x in new_id_to_connections[key] if x in nodes_in_subtree]

    for key in new_id_to_node:
        new_id_to_node[key] = stringify_node(new_id_to_node[key])

    return root_id, new_id_to_node, new_id_to_connections


def get_subtree_dfs(root_id, id_to_node, id_to_connections):
    stack = [root_id]
    subtree_nodes = set()

    while len(stack) != 0:
        top = stack.pop()
        if top not in subtree_nodes:
            subtree_nodes = subtree_nodes.union({top})

            if id_to_node[top].type != FeatureNode.IDENTIFIER_TOKEN or id_to_node[top].type != FeatureNode.TOKEN:
                for node_id in id_to_connections[top]:
                    stack.append(node_id)
    return subtree_nodes


# [{
#     'body': {
#         'id_to_node': {id -> nodeInfo},
#         'id_to_connections': {id -> [ids]},
#         'source': [tokens],
#         'root': 'body_node_id'
#     },
#     'name': name,
#     'type': return_type,
#     'javadoc': javadoc_string
# }]
def get_methods(id_to_node, id_to_connections):
    methods = []

    for node in id_to_node.values():
        if node.type == FeatureNode.FAKE_AST and node.contents == "MEMBERS":
            for connection_id in id_to_connections[node.id]:
                if id_to_node[connection_id].contents == "METHOD":
                    method_dict = {'parameters': []}

                    # extract return type of the method
                    for method_child in id_to_connections[connection_id]:
                        method_child_node = id_to_node[method_child]
                        if method_child_node.contents == "RETURN_TYPE":
                            method_dict['type'] = get_tokens(
                                id_to_node.values(),
                                method_child_node.startPosition,
                                method_child_node.endPosition
                            )

                        # extract parameters of the method (optional)
                        if method_child_node.contents == "PARAMETERS":
                            for parameter_child in id_to_connections[method_child]:
                                if id_to_node[parameter_child].contents == "VARIABLE":
                                    parameter = {}
                                    for variable_child in id_to_connections[parameter_child]:
                                        variable_child_node = id_to_node[variable_child]
                                        if variable_child_node.type == FeatureNode.IDENTIFIER_TOKEN:
                                            parameter['name'] = id_to_node[variable_child].contents
                                        elif variable_child_node.contents == "TYPE":
                                            parameter['type'] = get_tokens(
                                                id_to_node.values(),
                                                variable_child_node.startPosition,
                                                variable_child_node.endPosition
                                            )[0]
                                    if 'type' in parameter and 'name' in parameter:
                                        method_dict['parameters'].append(parameter)

                    if 'type' in method_dict:
                        for name_node in id_to_node.values():
                            # extract name
                            if connection_id in id_to_connections[name_node.id] \
                                    and name_node.type == FeatureNode.SYMBOL_MTH:
                                method_dict['name'] = name_node.contents.split("(")[0]

                            # extract javadoc and body AST
                            if connection_id in id_to_connections[name_node.id] \
                                    and name_node.type == FeatureNode.COMMENT_JAVADOC:
                                javadoc_comment = name_node.contents
                                javadoc_comment = clean_javadoc(javadoc_comment)
                                method_dict['javadoc'] = javadoc_comment

                                # since we found javadoc, we extract the body AST
                                for method_child in id_to_connections[connection_id]:
                                    method_child_node = id_to_node[method_child]
                                    if method_child_node.contents == "BODY":
                                        body_ast = {'root': method_child}

                                        _, new_id_to_node, new_id_to_connections = \
                                            get_subtree(method_child, id_to_node, id_to_connections)
                                        body_ast['id_to_node'] = new_id_to_node
                                        body_ast['id_to_connections'] = new_id_to_connections
                                        body_ast['source'] = get_tokens(
                                            id_to_node.values(),
                                            method_child_node.startPosition,
                                            method_child_node.endPosition
                                        )
                                        method_dict['body'] = body_ast
                        methods.append(method_dict)
    return methods


def update_statistics(source_dict, statistics):
    statistics['total_classes'] += 1
    statistics['methods'] += len(source_dict['methods'])
    statistics['fields'] += len(source_dict['fields'])

    for method in source_dict['methods']:
        if 'javadoc' in method:
            statistics['methods_with_javadoc'] += 1
            statistics['javadoc_length'] += len(method['javadoc'])
            if 'body' in method and 'source' in method['body']:
                statistics['javadoc_code_tokens'] += len(method['body']['source'])
                statistics['javadoc_code_characters'] += sum([len(x) for x in method['body']['source']])
                statistics['javadoc_ast_nodes'] += len(method['body']['id_to_node'])


def extract_corpus_features(rootdir):
    statistics = {
        'methods': 0,
        'fields': 0,
        'javadoc_code_tokens': 0,
        'javadoc_code_characters': 0,
        'javadoc_ast_nodes': 0,
        'javadoc_length': 0,
        'total_classes': 0,
        'methods_with_javadoc': 0
    }

    for root, _, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'rb') as f:
                g = Graph()
                print(path)
                g.ParseFromString(f.read())
                id_to_node, id_to_connections = graph_to_map(g)
                methods = get_methods(id_to_node, id_to_connections)
                fields = get_fields(id_to_node, id_to_connections)

                source_dict = {
                    'methods': methods,
                    'fields': fields,
                    'path': path
                }

                update_statistics(source_dict, statistics)

                with codecs.open(path + '.json', 'w', 'utf-8') as out:
                    json_dump = json.dumps(source_dict, ensure_ascii=False)
                    out.write(json_dump)

    print(statistics)
    return statistics


def parse_graph_and_run(path, method):
    with open(path, "rb") as f:
        g = Graph()
        g.ParseFromString(f.read())
        id_to_node, id_to_connections = graph_to_map(g)
        return method(id_to_node, id_to_connections)
