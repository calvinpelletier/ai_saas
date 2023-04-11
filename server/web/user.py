from collections import namedtuple
import hashlib
import string
import flask
import uuid

from ai_saas.lib.server import auth_fail


EMAIL_VALID_CHARACTERS = set(string.ascii_lowercase + string.digits + '@.-_')


User = namedtuple('User', [
    'uid',
    'email',
    'hash',
    'salt',
    'clearance',
    'iids',
    'cart',
])


def is_valid_email(email):
    return set(email) <= EMAIL_VALID_CHARACTERS and email.count('@') == 1


def hash_password(password, salt):
    return hashlib.sha512(
        password.encode('utf-8') + salt.encode('utf-8'),
    ).hexdigest()


class UserManager:
    def __init__(self, cfg, db, storage):
        self._min_signup_clearance = cfg.min_signup_clearance
        self._db_client = db

        self.token = _TokenManager(storage, 'tokens.csv')

        self._uid_to_user = {}
        self._email_to_user = {}


    def login(self, email, password):
        if self.is_signed_in():
            return 'already signed in'

        email = email.lower()
        if not is_valid_email(email):
            return 'invalid email'

        user = self.get_user_by_email(email)
        if user is None:
            return 'unknown email'

        if hash_password(password, user.salt) != user.hash:
            return 'incorrect password'

        self._login(user)
        return None


    def logout(self):
        if self.is_signed_in():
            flask.session.pop('uid')


    def signup(self, email, password):
        if self.is_signed_in():
            return 'already signed in'

        email = email.lower()
        if not is_valid_email(email):
            return 'invalid email'

        user = self.get_user_by_email(email)
        if user is not None:
            return 'email already in use'

        salt = uuid.uuid4().hex
        hash = hash_password(password, salt)
        clearance = self.token.get(default=0)

        user = self._add_user(email, hash, salt, clearance)

        self._login(user)
        return None


    def change_password(self, user, new_password):
        hash = hash_password(new_password, user.salt)
        resp = self._db_client.set_user_attr(user.uid, 'hash', hash)
        assert resp['success']

        user = self._load_user(uid=user.uid)
        assert user is not None


    def delete_user(self, user):
        resp = self._db_client.delete_user(user.uid)
        assert resp['success']

        self.logout()

        del self._uid_to_user[user.uid]
        del self._email_to_user[user.email]


    def can_signup(self):
        return self.get_clearance() >= self._min_signup_clearance


    def is_signed_in(self):
        return 'uid' in flask.session


    def get_user(self):
        if self.is_signed_in():
            uid = flask.session['uid']
            if uid in self._uid_to_user:
                return self._uid_to_user[uid]
            else:
                user = self._load_user(uid=uid)
                if user is not None:
                    return user
        return None


    def get_user_by_uid(self, uid):
        return self._uid_to_user[uid]


    def get_user_by_email(self, email):
        if email in self._email_to_user:
            return self._email_to_user[email]
        else:
            user = self._load_user(email=email)
            if user is not None:
                return user
        return None


    def get_clearance(self):
        user = self.get_user()
        if user is not None:
            return user.clearance
        return self.token.get(default=0)


    def refresh_user(self, user):
        return self._load_user(uid=user.uid)


    def _login(self, user):
        flask.session['uid'] = user.uid


    def _load_user(self, uid=None, email=None):
        resp = self._db_client.get_user(uid=uid, email=email)
        if resp['success']:
            user = User(**resp['user'])
            self._uid_to_user[user.uid] = user
            self._email_to_user[user.email] = user
            return user
        return None


    def _add_user(self, email, hash, salt, clearance):
        resp = self._db_client.add_user(email, hash, salt, clearance)
        assert resp['success']
        uid = resp['uid']
        user = self._load_user(uid=uid)
        assert user is not None
        return user


class _TokenManager:
    def __init__(self, storage, tokens_path):
        assert tokens_path.endswith('.csv')
        csv = storage.read(tokens_path)
        if csv is None:
            csv = ''
        self._tokens = {}
        for line in csv.split('\n'):
            if not len(line):
                continue
            token, clearance = line.strip().split(',')
            self._tokens[token] = int(clearance)


    def set(self, token):
        if token in self._tokens:
            flask.session['token'] = token


    def get(self, default=None):
        if 'token' in flask.session:
            return self._tokens[flask.session['token']]
        return default


    def clear(self):
        if 'token' in flask.session:
            flask.session.pop('token')
