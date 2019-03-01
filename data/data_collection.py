import json
import numpy as np
import os
import sys
import codecs
from re import finditer

from data.graph_pb2 import Graph
from data.graph_pb2 import FeatureNode
from interruptingcow import timeout


t_to_source = {
    'ABSTRACT': "abstract",
    'BREAK': "break",
    'CASE': "case",
    'CATCH': "catch",
    'CLASS': "class",
    'CONST': "const",
    'CONTINUE': "continue",
    'DEFAULT': "default",
    'DO': "do",
    'ELSE': "else",
    'EXTENDS': "extends",
    'FINAL': "final",
    'FINALLY': "finally",
    'FOR': "for",
    'GOTO': "goto",
    'IF': "if",
    'IMPLEMENTS': "implements",
    'IMPORT': "import",
    'INSTANCEOF': "instanceof",
    'INTERFACE': "interface",
    'NATIVE': "native",
    'NEW': "new",
    'PACKAGE': "package",
    'PRIVATE': "private",
    'PROTECTED': "protected",
    'PUBLIC': "public",
    'RETURN': "return",
    'STATIC': "static",
    'STRICTFP': "strictfp",
    'SWITCH': "switch",
    'SYNCHRONIZED': "synchronized",
    'THROW': "throw",
    'THROWS': "throws",
    'TRANSIENT': "transient",
    'TRY': "try",
    'VOLATILE': "volatile",
    'WHILE': "while",
    'ARROW': "->",
    'COLCOL': "::",
    'LPAREN': "(",
    'RPAREN': ")",
    'LBRACE': "{",
    'RBRACE': "}",
    'LBRACKET': "[",
    'RBRACKET': "]",
    'SEMI': ";",
    'COMMA': ",",
    'DOT': ".",
    'ELLIPSIS': "...",
    'EQ': "=",
    'GT': ">",
    'LT': "<",
    'BANG': "!",
    'TILDE': "~",
    'QUES': "?",
    'COLON': ":",
    'EQEQ': "==",
    'LTEQ': "<=",
    'GTEQ': ">=",
    'BANGEQ': "!=",
    'AMPAMP': "&&",
    'BARBAR': "||",
    'PLUSPLUS': "++",
    'SUBSUB': "--",
    'PLUS': "+",
    'SUB': "-",
    'STAR': "*",
    'SLASH': "/",
    'AMP': "&",
    'BAR': "|",
    'CARET': "^",
    'PERCENT': "%",
    'LTLT': "<<",
    'GTGT': ">>",
    'GTGTGT': ">>>",
    'PLUSEQ': "+=",
    'SUBEQ': "-=",
    'STAREQ': "*=",
    'SLASHEQ': "/=",
    'AMPEQ': "&=",
    'BAREQ': "|=",
    'CARETEQ': "^=",
    'PERCENTEQ': "%=",
    'LTLTEQ': "<<=",
    'GTGTEQ': ">>=",
    'GTGTGTEQ': ">>>=",
    'MONKEYS_AT': "@"
}

method_stats = {
    'qualified': 0,
    'total': 0
}


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


def to_original_source_code(source_tokens):
    result = []
    for token in source_tokens:
        if token in t_to_source:
            result.append(t_to_source[token])
        else:
            result.append(token)
    return result


def graph_to_map(graph):
    id_to_node = {}
    id_to_connections = {}

    for node in graph.node:
        id_to_node[node.id] = node
        id_to_connections[node.id] = set()

    for edge in graph.edge:
        id_to_connections[edge.sourceId].add(edge.destinationId)

    for key in id_to_connections:
        id_to_connections[key] = list(id_to_connections[key])

    return id_to_node, id_to_connections


def get_source(file, start_position, end_position):
    with open(file, 'rb') as f:
        file_str = f.read()
        return file_str


def get_tokens(nodes, string_nodes, start_position, end_position):
    tokens = []
    for node in nodes:
        if node.type == FeatureNode.IDENTIFIER_TOKEN or node.type == FeatureNode.TOKEN:
            if node.startPosition >= start_position and node.endPosition <= end_position:
                if node.id in string_nodes:
                    content = f"\"{node.contents}\""
                    tokens.append(content.encode("unicode_escape").decode("UTF-8"))
                else:
                    tokens.append(node.contents)
    return tokens


def get_string_nodes(id_to_node, id_to_connections):
    string_nodes = set()

    for node in id_to_node.values():
        if node.contents == "STRING_LITERAL":
            count = 0
            for con in id_to_connections[node.id]:
                con_node = id_to_node[con]
                if con_node.type in [FeatureNode.IDENTIFIER_TOKEN, FeatureNode.TOKEN] \
                        and con_node.contents not in t_to_source:
                    count += 1
                    string_nodes.add(con)
                    # if count > 1:
                    #     print("Something is wrong", [id_to_node[x].contents for x in id_to_connections[node.id]], node.id)
    return string_nodes


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
    string_nodes = get_string_nodes(id_to_node, id_to_connections)

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
                                string_nodes,
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
#     'javadoc': javadoc_string,
#     'parameters': [{'name': name, 'type': type}]
# }]
def get_methods(id_to_node, id_to_connections):
    methods = []
    string_nodes = get_string_nodes(id_to_node, id_to_connections)
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
                                string_nodes,
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
                                                string_nodes,
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
                                # print(name_node.contents, method_dict['type'])
                                method_dict['name'] = name_node.contents.split("(")[0]
                                if ">" in method_dict['name']:
                                    method_dict['name'] = method_dict['name'].split(">")[1]

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
                                            string_nodes,
                                            method_child_node.startPosition,
                                            method_child_node.endPosition
                                        )
                                        method_dict['body'] = body_ast
                        methods.append(method_dict)
    return methods


def update_statistics(source_dict, statistics):
    updated = False
    for method in source_dict['methods']:
        if 'javadoc' in method:
            if not updated:
                statistics['total_classes_with_javadoc'] += 1
                statistics['methods'] += len(source_dict['methods'])
                statistics['fields'] += len(source_dict['fields'])
                updated = True

            statistics['methods_with_javadoc'] += 1
            statistics['javadoc_length'] += len(method['javadoc'])
            if 'body' in method and 'source' in method['body']:
                statistics['javadoc_code_tokens'] += len(method['body']['source'])
                statistics['javadoc_code_characters'] += sum([len(x) for x in method['body']['source']])
                statistics['javadoc_ast_nodes'] += len(method['body']['id_to_node'])


def extract_concode_like_features(rootdir, seed):
    train = 0.8
    test = 0.1
    np.random.seed(seed)

    count = 0
    for root, _, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(root, file)
            if not path.endswith(".json"):
                with open(path, 'rb') as f:
                    try:
                        with timeout(60 * 5, exception=RuntimeError):
                            count += 1
                            g = Graph()
                            print(path)
                            g.ParseFromString(f.read())
                            # print(g)
                            id_to_node, id_to_connections = graph_to_map(g)
                            methods = get_methods(id_to_node, id_to_connections)
                            fields = get_fields(id_to_node, id_to_connections)

                            source_dict = {
                                'methods': methods,
                                'fields': fields,
                                'path': path
                            }

                            data = create_data(source_dict)

                            rnd = np.random.rand()
                            if rnd > train+test:
                                file_name = "valid"
                            elif rnd > train:
                                file_name = "test"
                            else:
                                file_name = "train"

                            with codecs.open(file_name + '.json', 'a', 'utf-8') as out:
                                for d in data:
                                    json_dump = json.dumps(d, ensure_ascii=False)
                                    out.write(json_dump)
                                    out.write("\n")
                    except RuntimeError:
                        pass


def create_data(class_json):
    methods_with_javadoc = []
    for method in class_json['methods']:
        if 'javadoc' in method:
            methods_with_javadoc.append(method)

    dataset = []
    for javadoc_method in methods_with_javadoc:
        if 'body' in javadoc_method:
            code = extract_code(javadoc_method)
            if len(code) < 500:
                method_stats['qualified'] += 1
                print(method_stats['qualified'])

                datapoint = {
                    'nl': extract_javadoc_from_method(javadoc_method),
                    'nlToks': extract_javadoc_from_method(javadoc_method).split(" "),
                    'memberVariables': extract_fields(class_json),
                    'memberFunctions': extract_methods(class_json, javadoc_method),
                    'code': code,
                    'renamed': rename_code(javadoc_method),
                    'repo': "no_repo",
                    "className": extract_class_name(class_json)
                }
                dataset.append(datapoint)
        method_stats['total'] += 1
        print("Total", method_stats['total'])
    return dataset


def split_camel_case(name):
    matches = finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', name)
    return [m.group(0) for m in matches]


def extract_class_name(class_json):
    path = class_json['path']
    name = path.split("/")[-1].split(".")[0]
    return name


def extract_code(javadoc_method):
    source = javadoc_method['body']['source']
    parameters = []
    if 'parameters' in javadoc_method:
        for parameter in javadoc_method['parameters']:
            string_value = [parameter['type']] + [parameter['name']]
            if len(parameters) == 0:
                parameters += string_value
            else:
                parameters += ([","] + string_value)
    method_string = javadoc_method['type'] + [javadoc_method['name']]
    result = method_string + ["LPAREN"] + parameters + ["RPAREN"] + source
    result = to_original_source_code(result)

    return result


def rename_code(javadoc_method):
    names = {}

    source = javadoc_method['body']['source']
    parameters = []
    if 'parameters' in javadoc_method:
        for parameter in javadoc_method['parameters']:
            new_name = "arg" + str(len(names))
            names[parameter['name']] = new_name
            string_value = [parameter['type']] + [new_name]
            if len(parameters) == 0:
                parameters += string_value
            else:
                parameters += ([","] + string_value)
    method_string = javadoc_method['type'] + ["function"]
    renamed_source = []
    for token in source:
        if token in names:
            renamed_source.append(names[token])
        else:
            renamed_source.append(token)

    result = method_string + ["LPAREN"] + parameters + ["RPAREN"] + source
    result = to_original_source_code(result)
    return result


def extract_fields(class_json):
    fields = class_json['fields']
    result = {}
    for field in fields:
        # print("Field", field)
        key = field['name'] + "=0"
        value = ''.join(field['type'])
        result[key] = value
    return result


def extract_methods(class_json, javadoc_method):
    methods = {}
    for method in class_json['methods']:
        if not('javadoc' in method and method['javadoc'] == javadoc_method['javadoc']):
            key = method['name']
            value = [''.join(method['type'])]

            if 'parameters' in method:
                for parameter in method['parameters']:
                    string_value = ''.join(parameter['type']) + " " + parameter['name']
                    value.append(string_value)

            methods[key] = [value]
    return methods


def extract_javadoc_from_method(javadoc_method):
    javadoc = javadoc_method['javadoc']
    return javadoc


def extract_corpus_features(rootdir):
    statistics = {
        'methods': 0,
        'fields': 0,
        'javadoc_code_tokens': 0,
        'javadoc_code_characters': 0,
        'javadoc_ast_nodes': 0,
        'javadoc_length': 0,
        'total_classes_with_javadoc': 0,
        'methods_with_javadoc': 0
    }

    for root, _, files in os.walk(rootdir):
        for file in files:
            path = os.path.join(root, file)
            with open(path, 'rb') as f:
                g = Graph()
                print(path)
                g.ParseFromString(f.read())
                # print(g)
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
