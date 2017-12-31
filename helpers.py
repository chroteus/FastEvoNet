import os, json

"""
    Relative string to module.
"""
def rel_path(*args):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), *args)

"""
    Decodes a string, encoded by Network.save
    into weights data for Network.
"""
def decode_net_str(data):
    return json.loads(data)


def open_net_file(net_file):
    path = rel_path(net_file)
    assert os.path.isfile(path)

    with open(path) as f:
        # separate weight and connection data into
        # separate chunks
        data = f.read().rstrip(os.linesep)

    return decode_net_str(data)
