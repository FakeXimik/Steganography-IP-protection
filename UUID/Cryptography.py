import json
import hashlib
import uuid
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from UserMetadata import UserMetadata

class User:
    # Class to simulate users + creates them unique ECC pairs

    def __init__(self, username: str):
        self.username = username
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key =  self.private_key.public_key()
    
    def __str__(self):
        return f"Username: {self.username} \nPublic key: {self.public_key}"


def proccess_and_sign_metadata(metadata_dict: dict, private_key) -> dict:
    """Parses a dictionary, generates a SHA-256 hash, and signs it using ECC."""

    # Serialize dict + sort to keep the cash consistent
    json_string = json.dumps(metadata_dict, sort_keys=True)
    data_bytes = json_string.encode('utf-8')

    # Generate SHA-256 hash
    sha256_hash = hashlib.sha256(data_bytes).hexdigest()

    # Generate the Digital Signature using ECC
    signature = private_key.sign(data_bytes, ec.ECDSA(hashes.SHA256()))

    return {
        "serialized_data": json_string,
        "sha256_hash": sha256_hash,
        "signature_hex": signature.hex() # Converted to hex for readable output
    }


def verify_signature(signature_hex, public_key, og_json_string):

    is_valid = False
    signature_bytes = bytes.fromhex(signature_hex) 
    data_bytes = og_json_string.encode('utf-8') # JSON text back into raw bytes

    try:
        # Verify signature
        public_key.verify(
            signature_bytes,
            data_bytes,
            ec.ECDSA(hashes.SHA256())
        )
        print("Signature is valid! The data is safe to use.")
        return True
        
    except InvalidSignature:
        print("Signature is invalid!")
        return False



def retrieve_data(og_json_string):
    recovered_dict = json.loads(og_json_string)
    print(f"The author is: {recovered_dict['author']}")
    print(f"The secret message is: {recovered_dict['data']}")


#---------------------------------------------------

if __name__ == "__main__":
    print ('---------------------------------------------------')
    user1 = User("Cool dude")
    print(user1)
    print ('---------------------------------------------------')
    metadata_dict1 = UserMetadata("Sam", "This data is protected because I said so.").get_metadata()
    print(metadata_dict1)
    print ('---------------------------------------------------')
signed_metadata = proccess_and_sign_metadata(metadata_dict1, user1.private_key)

print(signed_metadata)
print ('---------------------------------------------------')
is_valid = verify_signature(signed_metadata['signature_hex'], user1.public_key, signed_metadata['serialized_data'])
print ('---------------------------------------------------')
if (is_valid):
    retrieve_data(signed_metadata['serialized_data'])
print ('---------------------------------------------------')
print(uuid.uuid4())