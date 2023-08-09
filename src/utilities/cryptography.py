import hashlib


def hash_sha256(data):
    hash_object = hashlib.sha256()
    hash_object.update(data.encode())
    return hash_object.hexdigest()
