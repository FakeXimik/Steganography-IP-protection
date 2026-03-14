import json
import hashlib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec

def proccess_and_sign_metadata(metadata_dict: dict, private_key) -> dict:
    """Parses a dictionary, generates a SHA-256 hash and signs it using ECC"""

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
        "signature_hex": signature.hex() # Convert to hex for readable output
    }

def verify_signature(signature_hex, public_key, og_json_string):
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
    recovered_dict = json.loads(og_json_string) # parse a JSON string into obj
    print(f"The author is: {recovered_dict['author']}")
    print(f"The secret message is: {recovered_dict['data']}")

