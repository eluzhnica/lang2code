import codecs
import json
import os
import torch
from re import finditer

import numpy


# [{
#     'environment': {
#         'methods': [{
#             'name': "name",
#             'type': "type"
#         }],
#         'fields': [{
#             'name': "name",
#             'type': "type"
#         }]
#     },
#     'method': {
#         'doc': ["nl", "documentation", "tokens"],
#         'ast_flat': ['flattened ast nodes in depth first, left-to-right fashion'],
#         'ast': ['ast']
#     }
# }]
from data.graph_pb2 import FeatureNode
from model.layers import Encoder


def create_data(class_json, production_rule_results):
    methods_with_javadoc = []
    for method in class_json['methods']:
        if 'javadoc' in method:
            methods_with_javadoc.append(method)

    dataset = []
    for javadoc_method in methods_with_javadoc:
        if 'body' in javadoc_method:
            datapoint = {
                'method': {
                    'doc': extract_javadoc_from_method(javadoc_method),
                    'ast_flat': flatten_ast(javadoc_method['body'], production_rule_results),
                    'ast': javadoc_method['body']
                },
                'environment': {
                    'methods': extract_methods(class_json, javadoc_method),
                    'fields': extract_fields(class_json)
                }
            }
            dataset.append(datapoint)
    return dataset


def split_camel_case(name):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', name)
    return [m.group(0) for m in matches]


def extract_fields(class_json):
    result = class_json['fields']
    for field in result:
        field['type'] = ''.join(field['type'])
        field['name'] = split_camel_case(field['name'][0])
    return result


def extract_methods(class_json, javadoc_method):
    methods = []
    for method in class_json['methods']:
        if not('javadoc' in method and method['javadoc'] == javadoc_method['javadoc']):
            method_dict = {
                'name': split_camel_case(method['name'][0]),
                'type': ''.join(method['type'])
            }
            methods.append(method_dict)
    return methods


def extract_javadoc_from_method(javadoc_method):
    javadoc = javadoc_method['javadoc'].split(" ")
    return javadoc


def dfs_ast_left_to_right(id_to_node, id_to_connections, root_id, production_rules_result):
    stack = [root_id]
    visited = set()
    flattened = []

    while len(stack) != 0:
        top = stack.pop()
        if top not in visited:
            current_node = id_to_node[str(top)]
            flattened.append(current_node)
            visited.add(top)

            right_side = []
            left_side = current_node['type'] \
                if current_node['type'] in [FeatureNode.IDENTIFIER_TOKEN, FeatureNode.TOKEN] else current_node['contents']

            if production_rules_result is not None:
                for node_id in id_to_connections[str(top)]:
                    node = id_to_node[str(node_id)]
                    if node['type'] == FeatureNode.IDENTIFIER_TOKEN or node['type'] == FeatureNode.TOKEN:
                        right_side.append(node['type'])
                    else:
                        right_side.append(node['contents'])

                    if left_side in production_rules_result:
                        if right_side not in production_rules_result[left_side]:
                            production_rules_result[left_side].append(right_side)

            if current_node['type'] != FeatureNode.IDENTIFIER_TOKEN or current_node['type'] != FeatureNode.TOKEN:
                for node_id in id_to_connections[str(top)]:
                    stack.append(node_id)
    return flattened


def flatten_ast(body_dict, production_rules_result):
    id_to_node = body_dict['id_to_node']
    id_to_connections = body_dict['id_to_connections']
    root_id = body_dict['root']

    result = dfs_ast_left_to_right(id_to_node, id_to_connections, root_id, production_rules_result)

    processed_result = []
    for node in result:
        if node['type'] == FeatureNode.IDENTIFIER_TOKEN or node['type'] == FeatureNode.TOKEN:
            processed_result.append(node['type'])
            processed_result.append(node['contents'])
        else:
            processed_result.append(node['contents'])

    return processed_result


def types_names_dictionary(dataset):
    types = set()
    names = set()
    for data in dataset:
        if 'environment' in data:
            for method in data['environment']['methods']:
                types.add(method['type'])
                names = names.union(set(method['name']))
            for field in data['environment']['fields']:
                types.add(field['type'])
                names = names.union(set(field['name']))
        if 'method' in data:
            names = names.union(set(data['method']['doc']))

    # print(types)
    # print(names)
    types = {word: n for (n, word) in enumerate(types)}
    names = {word: n for (n, word) in enumerate(names)}
    return types, names


def prepare_data(dataset, name_dict, types_dict):
    for data in dataset:
        if 'environment' in data:
            for method in data['environment']['methods']:
                    method['name'] = torch.LongTensor([name_dict[x] for x in method['name']])
                    method['type'] = torch.LongTensor([types_dict[method['type']]])
            for field in data['environment']['fields']:
                if not isinstance(field['name'], torch.LongTensor):
                    field['name'] = torch.LongTensor([name_dict[x] for x in field['name']])
                    field['type'] = torch.LongTensor([types_dict[field['type']]])
        data['method']['doc'] = torch.LongTensor([name_dict[x] for x in data['method']['doc']])


def split_data(rootdir):
    js_files = []
    for root, _, files in os.walk(rootdir):
        for file in files[:2]:
            path = os.path.join(root, file)
            if path.endswith(".json"):
                with codecs.open(path, 'r', 'utf-8') as f:
                    js = json.loads(f.read())
                    js_files.append(js)
    train = js_files[len(js_files)-2000:]
    test = js_files[:len(js_files)-2000]

    production_rules = {}
    train_data = []
    test_data = []
    count = 0
    for class_json in train:
        print(count)
        count+=1
        train_data += create_data(class_json, production_rules)

    for class_json in test:
        print(count)
        count+=1
        test_data += create_data(class_json, None)

    print("Methods", len([x for x in train_data if 'method' in x]))

    types, names = types_names_dictionary(train_data)
    encoder = Encoder(types, 50, names, 25)
    prepare_data(train_data, names, types)
    for data in train_data:
        methods = torch.LongTensor([]) if 'environment' not in data else data['environment']['methods']
        fields = torch.LongTensor([]) if 'environment' not in data else data['environment']['fields']

        nl = data['method']['doc']
        encoder(nl, methods, fields)

