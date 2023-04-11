from time import sleep
import json

from ai_saas.lib.config import load_config
from ai_saas.lib.broker import Broker
from ai_saas.constants import Status
from ai_saas.lib.storage import StorageClient
from ai_saas.server.db.client import DatabaseClient


class Worker:
    def __init__(self, name, env, impl, needs_db=False):
        self.cfg = load_config(f'worker/{name}', env)
        self.impl = impl
        self.storage = StorageClient(self.cfg.storage.bucket)

        if needs_db:
            self.db = DatabaseClient(
                self.cfg.db.host,
                self.cfg.db.port,
                self.storage.read_or_init_key('internal'),
            )

        self.broker = Broker(
            self.cfg,
            use_compute_as_broker=True,
            storage=self.storage,
        )

    def pre(self, req):
        raise NotImplmentedError()

    def post(self, *args):
        raise NotImplmentedError()

    def run(self, iid):
        print(iid)
        prep = self.pre(iid)
        output = self.impl(*prep)
        self.post(iid, *output)
        success = True # TODO
        print(success)
        if success:
            if self.cfg.task.next:
                self.broker.add_to_task_q(self.cfg.task.next, iid)
            else:
                self.broker.set_status(self.cfg.task.job, iid, Status.DONE)
        else:
            self.broker.set_status(self.cfg.task.job, iid, Status.ERROR)

    def start(self):
        with self.impl.context():
            while True:
                req = self.broker.pop_from_task_q(self.cfg.task.name)
                if req:
                    self.run(req)
                else:
                    sleep(self.cfg.task.sleep)


class LiveWorker(Worker):
    def run(self, req):
        req = json.loads(req)
        prep = self.pre(req)
        output = self.impl(*prep)
        resp = self.post(*output)
        k = self.broker.get_result_key(self.cfg.task.name, req['iid'])
        self.broker.set(k, resp)
