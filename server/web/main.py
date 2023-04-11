import os
import flask
from marshmallow import Schema, fields
from functools import wraps
from PIL import Image
from time import time, sleep
import stripe
from base64 import b16encode

from ai_saas.lib.server import (run_dev_server, get_req, fail, success,
    auth_fail, validate_json, img_upload_schema, get_uploaded,
    validate_file_upload, Server, raw_json_resp, build_server)
from ai_saas.server.db.client import DatabaseClient
from ai_saas.server.compute.client import ComputeClient
from ai_saas.server.web.user import UserManager
from ai_saas.constants import ORIGINAL_IMG, FINAL_IMG, FINAL_IMG_WM, FINAL_THUMB
from ai_saas.lib.img import decode_img
from ai_saas.util import create_thumbnail
from ai_saas.lib.captcha import assess_captcha
from ai_saas.temp import (STRIPE_KEY, URL, PRICE_KEY, FULFILL_SIG, PROJECT_ID,
    CAPTCHA_KEY, CAPTCHA_ACTION, CAPTCHA_THRESHOLD)


stripe.api_key = STRIPE_KEY


def build(cfg, app):
    server = WebServer(cfg, app)

    @app.route('/', defaults={'value': ''})
    @app.route('/<path:value>')
    def _root_(value):
        static_dir = server.app.static_folder
        if value != '' and os.path.exists(os.path.join(static_dir, value)):
            return flask.send_from_directory(static_dir, value)
        return server._client_()

    @app.route('/signup', methods=['GET'])
    def signup():
        token = flask.request.args.get('a', default=None)
        return server.signup(token)

    @app.route('/logout', methods=['GET'])
    def logout():
        return server.logout()

    class WaitingListSchema(Schema):
        email = fields.Str(required=True)
        token = fields.Str(required=True)
    @app.route('/api/waitinglist/add', methods=['POST'])
    @validate_json(WaitingListSchema())
    def api_waitinglist_add():
        req = get_req()
        return server.api_waitinglist_add(req['email'], req['token'])

    class LoginSchema(Schema):
        email = fields.Str(required=True)
        password = fields.Str(required=True)
    @app.route('/api/auth/login', methods=['POST'])
    @validate_json(LoginSchema())
    def api_auth_login():
        req = get_req()
        return server.api_auth_login(req['email'], req['password'])

    class SignupSchema(Schema):
        email = fields.Str(required=True)
        password = fields.Str(required=True)
    @app.route('/api/auth/signup', methods=['POST'])
    @validate_json(SignupSchema())
    def api_auth_signup():
        req = get_req()
        return server.api_auth_signup(req['email'], req['password'])

    class ChangePassSchema(Schema):
        new_password = fields.Str(required=True)
    @app.route('/api/user/changepassword', methods=['POST'])
    @server._protected_()
    @validate_json(ChangePassSchema())
    def api_user_changepassword(user):
        req = get_req()
        return server.api_user_changepassword(user, req['new_password'])

    @app.route('/api/user/deleteaccount', methods=['POST'])
    @server._protected_()
    def api_user_deleteaccount(user):
        return server.api_user_deleteaccount(user)

    class GetImageSchema(Schema):
        iid = fields.Int(required=True)
    @app.route('/api/image/get', methods=['POST'])
    @server._protected_()
    def api_image_get(user):
        req = get_req()
        return server.api_image_get(user, int(req['iid']))

    @app.route('/api/image/get/all', methods=['POST'])
    @server._protected_()
    def api_image_get_all(user):
        return server.api_image_get_all(user)

    @app.route('/api/image/upload', methods=['POST'])
    @server._protected_()
    @validate_file_upload(img_upload_schema)
    def api_image_upload(user):
        return server.api_image_upload(user, get_uploaded())

    class DeleteImageSchema(Schema):
        iids = fields.Raw(required=True)
    @app.route('/api/image/delete', methods=['POST'])
    @server._protected_()
    @validate_json(DeleteImageSchema())
    def api_image_delete(user):
        req = get_req()
        iids = [int(x) for x in req['iids']]
        return server.api_image_delete(user, iids)

    class StatusSchema(Schema):
        iid = fields.Int(required=True)
    @app.route('/api/status/<job>', methods=['POST'])
    @server._protected_()
    @validate_json(StatusSchema())
    def api_status(user, job):
        if job not in ['upload', 'preprocess', 'postprocess']:
            return req_fail()
        req = get_req()
        return server.api_status(user, job, int(req['iid']))

    class SetStaticInputSchema(Schema):
        iid = fields.Int(required=True)
        si = fields.Raw(required=True)
    @app.route('/api/image/set/si', methods=['POST'])
    @server._protected_()
    @validate_json(SetStaticInputSchema())
    def api_image_set_si(user):
        req = get_req()
        return server.api_image_set_si(user, int(req['iid']), req['si'])

    class CalcPreviewSchema(Schema):
        iid = fields.Int(required=True)
        di = fields.Raw(required=True)
    @app.route('/api/calc/preview', methods=['POST'])
    @server._protected_()
    @validate_json(CalcPreviewSchema())
    def api_calc_preview(user):
        req = get_req()
        return server.api_calc_preview(user, int(req['iid']), req['di'])

    class CalcFinalSchema(Schema):
        iid = fields.Int(required=True)
        di = fields.Raw(required=True)
    @app.route('/api/calc/final', methods=['POST'])
    @server._protected_()
    @validate_json(CalcFinalSchema())
    def api_calc_final(user):
        req = get_req()
        return server.api_calc_final(user, int(req['iid']), req['di'])

    @app.route('/api/cart/get', methods=['POST'])
    @server._protected_()
    def api_cart_get(user):
        return server.api_cart_get(user)

    class CartAddSchema(Schema):
        iid = fields.Int(required=True)
    @app.route('/api/cart/add', methods=['POST'])
    @server._protected_()
    @validate_json(CartAddSchema())
    def api_cart_add(user):
        req = get_req()
        return server.api_cart_add(user, int(req['iid']))

    class CartRemoveSchema(Schema):
        iid = fields.Int(required=True)
    @app.route('/api/cart/remove', methods=['POST'])
    @server._protected_()
    @validate_json(CartRemoveSchema())
    def api_cart_remove(user):
        req = get_req()
        return server.api_cart_remove(user, int(req['iid']))

    @app.route('/api/cart/empty', methods=['POST'])
    @server._protected_()
    def api_cart_empty(user):
        return server.api_cart_empty(user)

    @app.route('/api/checkout', methods=['POST'])
    @server._protected_()
    def api_checkout(user):
        return server.api_checkout(user)

    @app.route('/api/fulfill', methods=['POST'])
    def api_fulfill():
        return server.api_fulfill()

    return server


class WebServer(Server):
    def __init__(self, cfg, app):
        super().__init__(cfg, app)

        self._compute = ComputeClient(
            cfg.compute.host,
            cfg.compute.port,
            self._internal_key,
        )

        self._db = DatabaseClient(cfg.db.host, cfg.db.port, self._internal_key)

        app.secret_key = self._storage.read_or_init_key('secret')
        self._um = UserManager(cfg, self._db, self._storage)


    def _client_(self):
        user = self._um.get_user()
        uid = user.uid if user else ''
        email = user.email if user else ''
        return flask.render_template('index.html', uid=uid, email=email)


    def _protected_(self, min_clearance=None):
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                user = self._um.get_user()
                if user is None:
                    return auth_fail()
                if min_clearance is not None and user.clearance < min_clearance:
                    return auth_fail()
                return f(user, *args, **kwargs)
            return wrapper
        return decorator


    def signup(self, token):
        if token:
            self._um.token.set(token)

        if self._um.can_signup():
            return flask.make_response(flask.redirect('/'))

        return self._client_()


    def logout(self):
        self._um.logout()
        self._um.token.clear()
        return flask.make_response(flask.redirect('/'))


    def api_waitinglist_add(self, email, token):
        score = assess_captcha(PROJECT_ID, CAPTCHA_KEY, token, CAPTCHA_ACTION)
        if score is None:
            return fail('captcha invalid')
        if score < CAPTCHA_THRESHOLD:
            return fail(f'captcha fail (score: {score})')

        resp = self._db.add_preuser(email)
        assert resp['success'], 'TODO'
        return success()


    def api_auth_login(self, email, password):
        error = self._um.login(email, password)
        if error:
            return fail(error)
        return success()


    def api_auth_signup(self, email, password):
        if not self._um.can_signup():
            return auth_fail()

        error = self._um.signup(email, password)
        if error:
            return fail(error)

        return success()


    def api_user_changepassword(self, user, new_password):
        self._um.change_password(user, new_password)
        return success()


    def api_user_deleteaccount(self, user):
        self._um.delete_user(user)
        # TODO: delete user images
        return success()


    def api_image_get(self, user, iid):
        if iid not in user.iids:
            return auth_fail()
        resp = self._db.get_image(iid)
        add_img_urls(self._storage, resp['image'])
        return resp


    def api_image_get_all(self, user):
        resp = self._db.get_images(user.iids)
        for image in resp['images']:
            add_img_urls(self._storage, image)
        return resp


    def api_image_upload(self, user, file):
        # validation
        img, error = validate_photo(file)
        if error:
            return fail(error)

        # create thumbnail
        thumb = create_thumbnail(img)

        # add to db
        resp = self._db.add_image(
            uid=user.uid,
            meta={
                'size': img.size,
                'upload_timestamp': int(time()),
            },
        )
        iid = resp['iid']

        # upload to storage
        self._storage.write_to_img_dir(iid, ORIGINAL_IMG, img)
        self._storage.write_to_img_dir(iid, 'og_thumb.png', thumb)

        # refresh cur user's iids
        user = self._um.refresh_user(user)
        assert user.iids[-1] == iid

        # run upload workloads
        resp = self._compute.initiate('upload', iid)
        assert resp['success'], 'TODO'

        # TODO: need a better way to handle this
        status = self._compute.status('upload', iid)['status']
        while status != 2:
            sleep(1)
            status = self._compute.status('upload', iid)['status']

        return success(iid=iid)


    def api_image_delete(self, user, iids):
        for iid in iids:
            resp = self._db.delete_image(uid=user.uid, iid=iid)
            assert resp['success']

        user = self._um.refresh_user(user)

        for iid in iids:
            assert iid not in user.iids
            assert iid not in user.cart

        # TODO: delete image from storage

        return success()


    def api_status(self, user, job, iid):
        if iid not in user.iids:
            return auth_fail()
        return self._compute.status(job, iid)


    def api_image_set_si(self, user, iid, si):
        if iid not in user.iids:
            return auth_fail()

        resp = self._db.get_image(iid)
        if not resp['success']:
            # TODO
            return resp

        error = validate_si(si, resp['image']['faces'])
        if error:
            return fail(error)

        resp = self._db.set_image_attr(iid, 'si', si)
        if not resp['success']:
            # TODO
            return resp

        resp = self._compute.initiate('preprocess', iid)
        if not resp['success']:
            # TODO
            return resp

        return success()


    def api_calc_preview(self, user, iid, di):
        if iid not in user.iids:
            return auth_fail()

        return raw_json_resp(self.app, self._compute.process(iid, di))


    def api_calc_final(self, user, iid, di):
        if iid not in user.iids:
            return auth_fail()

        resp = self._db.set_image_attr(iid, 'di', di)
        if not resp['success']:
            # TODO
            return resp

        resp = self._compute.initiate('postprocess', iid)
        if not resp['success']:
            # TODO
            return resp

        return success()


    def api_cart_get(self, user):
        return success(cart=user.cart)


    def api_cart_add(self, user, iid):
        if iid not in user.iids:
            return auth_fail()

        if iid in user.cart:
            return fail('already in cart')

        user.cart.append(iid)

        resp = self._db.set_user_attr(user.uid, 'cart', user.cart)
        user = self._um.refresh_user(user)
        return success(cart=user.cart)


    def api_cart_remove(self, user, iid):
        if iid not in user.cart:
            return fail('iid not in cart')

        user.cart.remove(iid)

        resp = self._db.set_user_attr(user.uid, 'cart', user.cart)
        user = self._um.refresh_user(user)
        return success(cart=user.cart)


    def api_cart_empty(self, user):
        user.cart.clear()
        resp = self._db.set_user_attr(user.uid, 'cart', user.cart)
        user = self._um.refresh_user(user)
        return success(cart=user.cart)


    def api_checkout(self, user):
        assert len(user.cart) > 0
        checkout_session = stripe.checkout.Session.create(
            line_items=[{
                'price': PRICE_KEY,
                'quantity': len(user.cart),
            }],
            mode='payment',
            success_url=URL + 'success?iids={}'.format(b16encode(
                ','.join([str(iid) for iid in user.cart]).encode('ascii'),
            ).decode('ascii')),
            cancel_url=URL,
            metadata={
                'uid': str(user.uid),
                'iids': ','.join([str(iid) for iid in user.cart]),
            },
        )
        return flask.redirect(checkout_session.url, code=303)


    def api_fulfill(self):
        payload = flask.request.get_data()
        sig = flask.request.headers['STRIPE_SIGNATURE']
        event = None
        try:
            event = stripe.Webhook.construct_event(payload, sig, FULFILL_SIG)
        except ValueError as e:
            return req_fail() # invalid payload
        except stripe.error.SignatureVerificationError as e:
            return req_fail() # invalid signature

        if event['type'] == 'checkout.session.completed':
            self._db.log_purchase(event)
            metadata = event['data']['object']['metadata']

            for iid in metadata['iids']:
                self._db.set_image_attr(iid, 'purchased', True)

            user = self._um.get_user_by_uid(metadata['uid'])
            user.cart.clear()
            resp = self._db.set_user_attr(user.uid, 'cart', user.cart)

        return success()


def validate_photo(file):
    # TODO: check size
    if file is None:
        return None, 'no file in request'
    try:
        img = Image.open(file)
    except:
        return None, 'could not open file'
    return img, None


def validate_si(si, faces):
    if len(si) != 2 or 'face' not in si or 'delta_g_sign' not in si:
        return 'invalid si'
    for face in faces:
        if face == si['face']:
            return None
    return 'face not in faces'


def add_img_urls(storage, data):
    img_urls = {}
    for label, fname in [
        ('og_full', 'og_full.png'),
        ('og_thumb', 'og_thumb.png'),
        ('final_full', FINAL_IMG if data['purchased'] else FINAL_IMG_WM),
        ('final_thumb', FINAL_THUMB),
        ('cna', 'cna.png'),
    ]:
        img_urls[label] = storage.signed_url_in_img_dir(data['iid'], fname)
    data['img_urls'] = img_urls


if __name__ == '__main__':
    # run_dev_server('web', build)
    server = build_server('web', 'staging', build)
    server._run_dev_()
