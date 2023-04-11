import json
from time import sleep

from ai_saas.lib.server import run_dev_server, req_fail, success, get_req, fail, \
    Server, raw_json_resp
from ai_saas.lib.broker import Broker


def build(cfg, app):
    server = ComputeServer(cfg, app)

    @app.route('/initiate', methods=['POST'])
    @server._internal_()
    def initiate():
        req = get_req()
        return server.initiate(req['job'], req['iid'])

    @app.route('/status', methods=['POST'])
    @server._internal_()
    def status():
        req = get_req()
        return server.status(req['job'], req['iid'])

    @app.route('/process', methods=['POST'])
    @server._internal_()
    def process():
        req = get_req()
        return server.process(req['iid'], req['di'])

    @app.route('/broker/rpush', methods=['POST'])
    @server._internal_()
    def broker_rpush():
        req = get_req()
        return server.broker_rpush(req['k'], req['v'])

    @app.route('/broker/lpop', methods=['POST'])
    @server._internal_()
    def broker_lpop():
        req = get_req()
        return server.broker_lpop(req['k'])

    @app.route('/broker/get', methods=['POST'])
    @server._internal_()
    def broker_get():
        req = get_req()
        return server.broker_get(req['k'])

    @app.route('/broker/set', methods=['POST'])
    @server._internal_()
    def broker_set():
        req = get_req()
        return server.broker_set(req['k'], req['v'])

    return server


class ComputeServer(Server):
    def __init__(self, cfg, app):
        super().__init__(cfg, app)
        self._broker = Broker(cfg)


    def initiate(self, job, iid):
        q_size = self._broker.run_job(
            job,
            getattr(self.cfg.job, job).first_task,
            iid,
        )
        return success(q_size=q_size)


    def status(self, job, iid):
        return success(status=self._broker.get_status(job, iid))


    def process(self, iid, di):
        self._broker.add_to_task_q('process', json.dumps({
            'iid': iid,
            'di': di,
        }))

        k = self._broker.get_result_key('process', iid)
        while True:
            img_b64 = self._broker.get(k)
            if img_b64 is not None:
                break
            sleep(self.cfg.live.process.sleep)
        self._broker.delete(k)

        return raw_json_resp(self.app, img_b64)


    def broker_rpush(self, k, v):
        self._broker._broker.rpush(k, v)
        return raw_json_resp(self.app, '')

    def broker_lpop(self, k):
        return raw_json_resp(self.app, self._broker._broker.lpop(k))

    def broker_get(self, k):
        return raw_json_resp(self.app, self._broker._broker.get(k))

    def broker_set(self, k, v):
        self._broker._broker.set(k, v)
        return raw_json_resp(self.app, '')


if __name__ == '__main__':
    run_dev_server('compute', build)
