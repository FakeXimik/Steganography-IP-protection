import os
from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

class User:
    """ Class to simulate users + creates or loads ENCRYPTED unique ECC pairs"""

    def __init__(self, username: str, password: bytes):
        self.username = username
        self.password = password
        self.key_filename = f"{username}_private_key.pem"
        
        self.home_dir = os.path.expanduser("~")
        self.steg_dir = os.path.join(self.home_dir, '.stegAI')
        self.pem_file_path = os.path.join(self.steg_dir, self.key_filename)

        os.makedirs(self.steg_dir, exist_ok=True)

        if os.path.exists(self.pem_file_path):
            print(f"Sup, {username}! Decrypting your saved private key...")
            self.private_key = self._load_private_key()
        else:
            print(f"New user: {username}! Generating and encrypting a new key...")
            self.private_key = ec.generate_private_key(ec.SECP256R1())
            self._save_private_key()

        self.public_key = self.private_key.public_key()
    
    def _save_private_key(self):
        """Saves the private key locked with a password"""

        pem_bytes = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(self.password) 
        )
        with open(self.pem_file_path, 'wb') as f:
            f.write(pem_bytes)

    def _load_private_key(self):
        """Loads and unlocks the private key using the password"""

        with open(self.pem_file_path, 'rb') as key_file:
            private_key = serialization.load_pem_private_key(
                key_file.read(),
                password=self.password,
            )
        return private_key

    def __str__(self):
        return f"Username: {self.username} \nPublic key: {self.public_key}"
