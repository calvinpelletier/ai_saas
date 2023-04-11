import requests


class InternalClient:
    def __init__(self, host, port, internal_key):
        self._host = f'http://{host}:{port}'
        self._internal_key = internal_key

    def _post(self, path, data={}, decode_json=True):
        data['_token_'] = self._internal_key
        resp = requests.post(self._host + path, json=data)
        if decode_json:
            resp = resp.json()
        return resp
