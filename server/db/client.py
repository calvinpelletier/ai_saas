from ai_saas.lib.client import InternalClient


class DatabaseClient(InternalClient):
    def add_preuser(self, email):
        return self._post('/preuser/add', {'email': email})

    def get_user(self, uid=None, email=None):
        assert uid is not None or email is not None
        return self._post('/user/get', {'uid': uid, 'email': email})

    def add_user(self, email, hash, salt, clearance):
        return self._post('/user/add', {
            'email': email,
            'hash': hash,
            'salt': salt,
            'clearance': clearance,
        })

    def delete_user(self, uid):
        return self._post('/user/delete', {'uid': uid})

    def set_user_attr(self, uid, attr, value):
        return self._post(f'/user/set/{attr}', {'uid': uid, 'value': value})

    def get_image(self, iid):
        return self._post('/image/get', {'iid': iid})

    def get_images(self, iids):
        return self._post('/image/get/multi', {'iids': iids})

    def add_image(self, uid, meta):
        return self._post('/image/add', {'uid': uid, 'meta': meta})

    def delete_image(self, uid, iid):
        return self._post('/image/delete', {'uid': uid, 'iid': iid})

    def set_image_attr(self, iid, attr, value):
        return self._post(f'/image/set/{attr}', {'iid': iid, 'value': value})

    def log_purchase(self, info):
        return self._port('/purchase/add', {'info': info})
