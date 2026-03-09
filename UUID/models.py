from datetime import datetime
from cryptography.hazmat.primitives.asymmetric import ec

class User:
    """ Class to simulate users + creates unique ECC pairs for them"""

    def __init__(self, username: str):
        self.username = username
        self.private_key = ec.generate_private_key(ec.SECP256R1())
        self.public_key =  self.private_key.public_key()
    
    def __str__(self):
        return f"Username: {self.username} \nPublic key: {self.public_key}"



class UserMetadata:
    """Class to generate and store metadata"""

    def __init__(self, author: str, text: str):
        self.author = author
        self.data = text
        self.created_at = datetime.now().isoformat() 
        self.copyright_info = f"Copyright {datetime.now().year}, All Rights Reserved"

    def get_metadata(self) -> dict:
        """Returns the generated metadata as a dictionary"""

        return {
            "author": self.author,
            "creation_date": self.created_at,
            "data": self.data
        }
    
    def __str__(self):
        return f"Here is basic metadata:\n Author: {self.author}\n Creation Date: {self.created_at}\n Data: {self.data}\n Copyright: {self.copyright_info}"
    
