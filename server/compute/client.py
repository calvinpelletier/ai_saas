from ai_saas.lib.client import InternalClient


class ComputeClient(InternalClient):
    def initiate(self, job, iid):
        return self._post('/initiate', {'job': job, 'iid': iid})

    def status(self, job, iid):
        return self._post('/status', {'job': job, 'iid': iid})

    def process(self, iid, di):
        return self._post('/process', {'iid': iid, 'di': di}, decode_json=False)


class ComputeAsBrokerClient(InternalClient):
    def rpush(self, k, v):
        return self._post('/broker/rpush', {
            'k': k,
            'v': v,
        }, decode_json=False).text.encode('utf-8')

    def lpop(self, k):
        return self._post('/broker/lpop', {
            'k': k,
        }, decode_json=False).text.encode('utf-8')

    def get(self, k):
        return self._post('/broker/get', {
            'k': k,
        }, decode_json=False).text.encode('utf-8')

    def set(self, k, v):
        return self._post('/broker/set', {
            'k': k,
            'v': v,
        }, decode_json=False).text.encode('utf-8')
