import flask
from functools import wraps
import os
from marshmallow import Schema, fields, validates_schema, ValidationError

from ai_saas.lib.config import load_config
from ai_saas.constants import CODE_PATH
from ai_saas.lib.storage import StorageClient


class Server:
    def __init__(self, cfg, app):
        self.cfg = cfg
        self.app = app
        self._storage = StorageClient(cfg.storage.bucket)
        self._internal_key = self._storage.read_or_init_key('internal')

    def _run_dev_(self):
        self.app.run(
            host='127.0.0.1',
            port=self.cfg.server.port,
            debug=True,
        )

    def _internal_(self):
        def decorator(f):
            @wraps(f)
            def wrapper(*args, **kwargs):
                req = get_req()
                if req is None or '_token_' not in req or \
                        req['_token_'] != self._internal_key:
                    return auth_fail()
                return f(*args, **kwargs)
            return wrapper
        return decorator


class _ImgUploadSchema(Schema):
    file = fields.Field(required=True)

    @validates_schema
    def validate_file(self, data, **kwargs):
        if 'file' not in data:
            return ValidationError('0')
        file = data['file']
        if not file or not file.filename or '.' not in file.filename:
            return ValidationError('1')
        extension = file.filename.rsplit('.', 1)[1].lower()
        if extension not in ['png', 'jpg', 'jpeg']:
            return ValidationError('2')

img_upload_schema = _ImgUploadSchema()


def build_server(name, env, builder):
    cfg = load_config(f'server/{name}', env)

    app = flask.Flask(
        __name__,
        root_path=os.path.join(CODE_PATH, f'server/{name}'),
        static_url_path='/static',
    )

    return builder(cfg, app)


def run_dev_server(name, builder):
    server = build_server(name, 'dev', builder)
    server._run_dev_()


def get_req():
    return flask.request.get_json()


def raw_json_resp(app, data):
    return app.response_class(
        response=data,
        status=200,
        mimetype='application/json'
    )


def get_uploaded():
    return flask.request.files['file']


def success(**resp):
    assert 'success' not in resp
    resp['success'] = True
    return flask.jsonify(resp), 200


def fail(error, code=200):
    return flask.jsonify({'success': False, 'error': error}), code


def auth_fail():
    return fail('unauthorized request', 401)


def req_fail():
    return fail('invalid request', 400)


def validate_json(schema):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            val_errors = schema.validate(flask.request.json)
            if val_errors:
                return req_fail()
            return f(*args, **kwargs)
        return wrapper
    return decorator


def validate_file_upload(schema):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            val_errors = schema.validate(flask.request.files)
            if val_errors:
                return req_fail()
            return f(*args, **kwargs)
        return wrapper
    return decorator
