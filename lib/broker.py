import redis

from ai_saas.constants import Status
from ai_saas.server.compute.client import ComputeAsBrokerClient


class Broker:
    def __init__(self, cfg, use_compute_as_broker=False, storage=None):
        if not use_compute_as_broker:
            self._broker = redis.Redis(
                host=cfg.broker.host,
                port=cfg.broker.port,
                db=cfg.broker.db,
            )
        else:
            self._broker = ComputeAsBrokerClient(
                cfg.compute.host,
                cfg.compute.port,
                storage.read_or_init_key('internal'),
            )

    def run_job(self, job, first_task, iid):
        self.set_status(job, iid, Status.IN_PROGRESS)
        return self.add_to_task_q(first_task, iid)

    def add_to_task_q(self, task, req):
        return self._broker.rpush(task, req)

    def pop_from_task_q(self, task):
        req = self._broker.lpop(task)
        return req.decode('utf-8') if req is not None else None

    def get_status(self, job, iid):
        status = self._broker.get(self.get_result_key(job, iid))
        return int(status) if status is not None else None

    def set_status(self, job, iid, status):
        self._broker.set(self.get_result_key(job, iid), int(status))

    def get(self, k):
        return self._broker.get(k)

    def set(self, k, v):
        return self._broker.set(k, v)

    def delete(self, k):
        self._broker.delete(k)

    def get_result_key(self, name, iid):
        return f'{name}/{iid}'
