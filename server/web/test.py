import os
from io import BytesIO
from time import sleep

from ai_saas.lib.dep import download_img
from ai_saas.lib.server import build_server
from ai_saas.server.web.main import build as build_webserver
from ai_saas.lib.img import decode_img


INPUT_PATH = download_img('sera13.jpg')
FACE_IDX = 1
DELTA_G_SIGN = -1


def print_resp(label, resp, json_resp=True):
    if json_resp:
        print(f'({label})', resp, resp.json, '\n')
    else:
        print(f'({label})', resp, '\n')


def wait_for(job):
    while True:
        resp = c.post(f'/api/status/{job}', json={'iid': iid})
        print_resp('status', resp)
        if resp.json['status'] != 1:
            break
        sleep(1)
    print('STATUS', resp.json['status'], '\n')


server = build_server('web', 'staging', build_webserver)
server.app.config.update({'TESTING': True})
c = server.app.test_client()

resp = c.post('/api/waitinglist/add', json={
    'email': 'test1@test.com',
    'token': 'asdf',
})
print_resp('waiting list (bad token)', resp)

signup_data = {
    'email': 'test@test.com',
    'password': 'test',
}
resp = c.post('/api/auth/signup', json=signup_data)
print_resp('unauthorized signup', resp)

resp = c.get('/signup?a=TpDW4unP')
print_resp('get signup page', resp, json_resp=False)

resp = c.post('/api/auth/signup', json=signup_data)
print_resp('valid signup', resp)

c.get('/logout')

resp = c.post('/api/auth/login', json=signup_data)
print_resp('login', resp)

new_pass = 'test2'
resp = c.post('/api/user/changepassword', json={'new_password': new_pass})
print_resp('change pass', resp)

c.get('/logout')

resp = c.post('/api/auth/login', json=signup_data)
print_resp('login using old password', resp)

login_data = {
    'email': signup_data['email'],
    'password': new_pass,
}
resp = c.post('/api/auth/login', json=login_data)
print_resp('login using new password', resp)

resp = c.post('/api/image/get/all')
print_resp('library', resp)

resp = c.post('/api/cart/get')
print_resp('cart', resp)

resp = c.post('/api/cart/add', json={'iid': 0})
print_resp('cart add', resp)

resp = c.post('/api/cart/get')
print_resp('cart', resp)

with open(INPUT_PATH, 'rb') as f:
    img_bytes = f.read()
resp = c.post('/api/image/upload', data={
    'file': (BytesIO(img_bytes), 'test.jpg'),
}, content_type='multipart/form-data')
print_resp('upload', resp)

resp = c.post('/api/image/get/all')
print_resp('library', resp)
iid = resp.json['images'][-1]['iid']

resp = c.post('/api/image/get', json={'iid': iid})
print_resp('last uploaded', resp)

wait_for('upload')

resp = c.post('/api/image/get', json={'iid': iid})
faces = resp.json['image']['faces']
print('faces', faces, '\n')

resp = c.post('/api/image/set/si', json={
    'iid': iid,
    'si': {
        'face': faces[FACE_IDX],
        'delta_g_sign': DELTA_G_SIGN,
    },
})
print_resp('set si', resp)

wait_for('preprocess')

resp = c.post('/api/calc/preview', json={
    'iid': iid,
    'di': {
        'delta_g_mag': 0.5,
    },
})
print_resp('process', resp, json_resp=False)

resp = c.post('/api/calc/final', json={
    'iid': iid,
    'di': {
        'delta_g_mag': 1.0,
    },
})
print_resp('calc final', resp)

wait_for('postprocess')

resp = c.post('/api/image/get/all')
print_resp('library', resp)

resp = c.post('/api/cart/get')
print_resp('cart', resp)

resp = c.post('/api/cart/add', json={'iid': iid})
print_resp('cart add', resp)

resp = c.post('/api/cart/get')
print_resp('cart', resp)

resp = c.post('/api/cart/remove', json={'iid': iid})
print_resp('cart remove', resp)

resp = c.post('/api/cart/get')
print_resp('cart', resp)

resp = c.post('/api/cart/add', json={'iid': iid})
print_resp('cart add', resp)

resp = c.post('/api/cart/empty')
print_resp('cart empty', resp)

resp = c.post('/api/cart/get')
print_resp('cart', resp)

resp = c.post('/api/cart/add', json={'iid': iid})
print_resp('cart add', resp)

resp = c.post('/api/checkout')
print_resp('checkout', resp)

resp = c.post('/api/image/delete', json={'iids': [iid]})
print_resp('delete image', resp)

resp = c.post('/api/image/get/all')
print_resp('library', resp)

resp = c.post('/api/cart/get')
print_resp('cart', resp)

resp = c.post('/api/user/deleteaccount')
print_resp('delete account', resp)

resp = c.post('/api/image/get/all')
print_resp('protected endpoint when not logged in', resp)

resp = c.post('/api/auth/login', json=login_data)
print_resp('login to deleted account', resp)
