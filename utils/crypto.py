# crypto.py
import json
import hashlib
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

def verify_signature(signature_hex: str, public_key_hex: str, og_json_string: str) -> bool:
    """Verifies a signature using raw hex strings provided by the TypeScript frontend."""
    try:
        signature_bytes = bytes.fromhex(signature_hex) 
        data_bytes = og_json_string.encode('utf-8') 
        public_key_bytes = bytes.fromhex(public_key_hex)
        
        # Load the raw bytes into a python cryptography object
        public_key = serialization.load_der_public_key(public_key_bytes)
        
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
    except Exception as e:
        print(f"Cryptographic Parsing Error: {e}")
        return False