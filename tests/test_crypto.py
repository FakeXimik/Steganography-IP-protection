# import pytest
# from cryptography.hazmat.primitives.asymmetric import ec
# from utils.crypto import *

# @pytest.fixture
# def dummy_key():
#     return ec.generate_private_key(ec.SECP256R1())

# @pytest.fixture
# def dummy_public_key(dummy_key):
#     return dummy_key.public_key()

# @pytest.fixture
# def dummy_data():
#     return {"author": "Sam", "data": "Secret"}

# @pytest.fixture
# def dummy_json():
#     return {"author": "Sam", "data": "Secret"}

# def test_proccess_and_sign_metadata(dummy_key, dummy_data):
#     result = proccess_and_sign_metadata(dummy_data, dummy_key)

#     assert type(result) is dict
#     assert "serialized_data" in result
#     assert "sha256_hash" in result
#     assert "signature_hex" in result
#     assert len(result) == 3

# def test_verify_signature_valid(dummy_key, dummy_public_key, dummy_data):
#     signed = proccess_and_sign_metadata(dummy_data, dummy_key)

#     is_valid = verify_signature(
#         signed["signature_hex"], 
#         dummy_public_key, 
#         signed["serialized_data"]
#     )
    
#     assert is_valid == True

# def test_verify_signature_invalid(dummy_key, dummy_public_key, dummy_data):
#     signed = proccess_and_sign_metadata(dummy_data, dummy_key)
#     tampered_json = '{"author": "Hacker", "data": "Secret"}'

#     is_valid = verify_signature(
#         signed["signature_hex"], 
#         dummy_public_key, 
#         tampered_json
#     )
    
#     assert is_valid == False

# def test_retrieve_data(capsys):
#     json_string = '{"author": "Sam", "data": "Secret"}'
    
#     retrieve_data(json_string)
    
#     # capsys reads what was printed to the terminal
#     captured = capsys.readouterr()
    
#     assert "The author is: Sam" in captured.out
#     assert "The secret message is: Secret" in captured.out


