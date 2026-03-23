import os
import pytest
from unittest.mock import patch
from cryptography.hazmat.primitives.asymmetric import ec
from models.user import User
from models.userMetadata import UserMetadata

# --- FIXTURES ---

@pytest.fixture
def mock_home(tmp_path):
    """
    Tricks the User class into saving the .pem files into a temporary, 
    disposable folder instead of your actual computer's home directory.
    """
    with patch("os.path.expanduser", return_value=str(tmp_path)):
        yield tmp_path

# --- TESTS FOR USER CLASS ---

def test_user_creation_and_save(mock_home):
    # 1. Initialize a brand new user
    user = User(username="Alice", password=b"super_secret_123")
    
    # 2. Verify keys were generated in memory
    assert isinstance(user.private_key, ec.EllipticCurvePrivateKey)
    assert isinstance(user.public_key, ec.EllipticCurvePublicKey)
    
    # 3. Verify the physical file was created in our isolated folder
    expected_file_path = os.path.join(mock_home, ".stegAI", "Alice_private_key.pem")
    assert os.path.exists(expected_file_path)

def test_user_load_existing_key(mock_home):
    # 1. Create a user to generate and save the file
    user1 = User(username="Bob", password=b"password123")
    
    # 2. Re-instantiate the exact same user. The class should realize the file 
    # exists and load it instead of creating a new one.
    user2 = User(username="Bob", password=b"password123")
    
    # 3. Mathematically prove both instances hold the exact same private key
    val1 = user1.private_key.private_numbers().private_value
    val2 = user2.private_key.private_numbers().private_value
    assert val1 == val2

def test_user_invalid_password(mock_home):
    # 1. Create a user
    User(username="Charlie", password=b"correct_password")
    
    # 2. Try to load the user with the wrong password
    # The cryptography library should violently throw a ValueError.
    with pytest.raises(ValueError):
        User(username="Charlie", password=b"wrong_password")

# --- TESTS FOR USERMETADATA CLASS ---

def test_user_metadata_generation():
    author_name = "SystemAdmin"
    secret_text = "This is highly classified."
    
    # Initialize class
    metadata_obj = UserMetadata(author=author_name, text=secret_text)
    
    # Generate the dictionary
    result = metadata_obj.get_metadata()
    
    # Verify the dictionary structure and contents
    assert isinstance(result, dict)
    assert result["author"] == author_name
    assert result["data"] == secret_text
    
    # Ensure the ISO format timestamp was actually generated
    assert "creation_date" in result
    assert len(result["creation_date"]) > 10

def test_user_metadata_string_format():
    metadata_obj = UserMetadata(author="Dave", text="Testing str")
    string_output = str(metadata_obj)
    
    # Verify the __str__ override actually prints all the relevant data
    assert "Dave" in string_output
    assert "Testing str" in string_output
    assert "Copyright" in string_output