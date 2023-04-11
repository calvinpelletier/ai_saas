from copy import deepcopy
from pathlib import Path

from ai_saas.server.db.database import get_db, PreUser, User, Image
from ai_saas.lib.server import run_dev_server, req_fail, success, get_req, fail, \
    Server
from ai_saas.constants import Status


def build(cfg, app):
    server = DatabaseServer(cfg, app)

    @app.route('/preuser/add', methods=['POST'])
    @server._internal_()
    def preuser_add():
        req = get_req()
        return server.preuser_add(req['email'])

    @app.route('/user/get', methods=['POST'])
    @server._internal_()
    def user_get():
        req = get_req()
        return server.user_get(req['uid'], req['email'])

    @app.route('/user/add', methods=['POST'])
    @server._internal_()
    def user_add():
        req = get_req()
        return server.user_add(
            req['email'],
            req['hash'],
            req['salt'],
            req['clearance'],
        )

    @app.route('/user/delete', methods=['POST'])
    @server._internal_()
    def user_delete():
        req = get_req()
        return server.user_delete(req['uid'])

    @app.route('/user/set/<attr>', methods=['POST'])
    @server._internal_()
    def user_set_attr(attr):
        req = get_req()
        return server.user_set_attr(req['uid'], attr, req['value'])

    @app.route('/image/get', methods=['POST'])
    @server._internal_()
    def image_get():
        req = get_req()
        return server.image_get(req['iid'])

    @app.route('/image/get/multi', methods=['POST'])
    @server._internal_()
    def image_get_multi():
        req = get_req()
        return server.image_get_multi(req['iids'])

    @app.route('/image/add', methods=['POST'])
    @server._internal_()
    def image_add():
        req = get_req()
        return server.image_add(req['uid'], req['meta'])

    @app.route('/image/delete', methods=['POST'])
    @server._internal_()
    def image_delete():
        req = get_req()
        return server.image_delete(req['uid'], req['iid'])

    @app.route('/image/set/<attr>', methods=['POST'])
    @server._internal_()
    def image_set_attr(attr):
        req = get_req()
        return server.image_set_attr(req['iid'], attr, req['value'])

    @app.route('/purchase/add', methods=['POST'])
    @server._internal_()
    def purchase_add():
        req = get_req()
        return server.purchase_add(req['info'])

    return server


class DatabaseServer(Server):
    def __init__(self, cfg, app):
        super().__init__(cfg, app)

        self.db = get_db()
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/test.db'
        app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
        self.db.init_app(app)
        with app.app_context():
            self.db.create_all() # create tables if they dont exist


    def preuser_add(self, email):
        preuser = PreUser.query.filter_by(email=email).first()
        if preuser is None:
            preuser = PreUser(email=email)
            self.db.session.add(preuser)
            self.db.session.commit()
        return success()


    def user_get(self, uid, email):
        if uid is not None:
            user = User.query.get(uid)
        elif email is not None:
            user = User.query.filter_by(email=email).first()
        else:
            return req_fail()

        if user is None:
            return fail('unknown user')

        return success(user=user.serialize())


    def user_add(self, email, hash, salt, clearance):
        user = User(
            email=email,
            hash=hash,
            salt=salt,
            clearance=clearance,
            iids=[],
            cart=[],
        )
        self.db.session.add(user)
        self.db.session.commit()
        return success(uid=user.uid)


    def user_delete(self, uid):
        user = User.query.get(uid)
        if user is None:
            return fail('unknown user')

        self.db.session.delete(user)
        self.db.session.commit()
        return success()


    def user_set_attr(self, uid, attr, value):
        user = User.query.get(uid)
        if user is None:
            return fail('unknown user')

        if not hasattr(user, attr):
            return fail('unknown attr')

        setattr(user, attr, value)

        self.db.session.commit()
        return success()


    def image_get(self, iid):
        return success(image=Image.query.get(iid).serialize())


    def image_get_multi(self, iids):
        images = [Image.query.get(iid).serialize() for iid in iids]
        return success(images=images)


    def image_add(self, uid, meta):
        user = User.query.get(uid)
        if user is None:
            return fail('unknown user')

        image = Image(
            uid=uid,
            meta=meta,
            faces=None,
            preprocessed=Status.NOT_STARTED,
            purchased=False,
            si=None,
            di=None,
        )
        self.db.session.add(image)
        self.db.session.commit()

        user.iids = user.iids + [image.iid] # TODO: better way to do this?
        self.db.session.commit()

        return success(iid=image.iid)


    def image_delete(self, uid, iid):
        user = User.query.get(uid)
        if user is None:
            return fail('unknown user')

        iids = deepcopy(user.iids)
        iids.remove(iid)
        user.iids = iids

        if iid in user.cart:
            cart = deepcopy(user.cart)
            cart.remove(iid)
            user.cart = cart

        self.db.session.commit()
        return success()


    def image_set_attr(self, iid, attr, value):
        image = Image.query.get(iid)
        if image is None:
            return fail('unknown image')

        if not hasattr(image, attr):
            return fail('unknown attr')

        setattr(image, attr, value)

        self.db.session.commit()
        return success()


    def purchase_add(self, info):
        purchase = Purchase(info=info)
        self.db.session.add(purchase)
        self.db.session.commit()
        return success()


if __name__ == '__main__':
    run_dev_server('db', build)
