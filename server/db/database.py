from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()
def get_db():
    return db


class PreUser(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(128), unique=True, nullable=False)


class Purchase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    info = db.Column(db.PickleType, nullable=False)


class User(db.Model):
    uid = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(128), unique=True, nullable=False)
    hash = db.Column(db.String(128), nullable=False)
    salt = db.Column(db.String(32), nullable=False)
    clearance = db.Column(db.Integer, nullable=False)
    iids = db.Column(db.PickleType, nullable=False)
    cart = db.Column(db.PickleType, nullable=False)

    def serialize(self):
        return {
            'uid': self.uid,
            'email': self.email,
            'hash': self.hash,
            'salt': self.salt,
            'clearance': self.clearance,
            'iids': self.iids,
            'cart': self.cart,
        }


class Image(db.Model):
    iid = db.Column(db.Integer, primary_key=True)
    uid = db.Column(db.Integer, nullable=False)
    meta = db.Column(db.PickleType, nullable=False)
    faces = db.Column(db.PickleType, nullable=True)
    preprocessed = db.Column(db.Integer, nullable=False)
    purchased = db.Column(db.Boolean, nullable=False)
    si = db.Column(db.PickleType, nullable=True)
    di = db.Column(db.PickleType, nullable=True)

    def serialize(self):
        return {
            'iid': self.iid,
            'uid': self.uid,
            'meta': self.meta,
            'faces': self.faces,
            'preprocessed': self.preprocessed,
            'purchased': self.purchased,
            'si': self.si,
            'di': self.di,
        }
