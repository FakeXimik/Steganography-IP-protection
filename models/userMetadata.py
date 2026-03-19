from datetime import datetime

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
    
