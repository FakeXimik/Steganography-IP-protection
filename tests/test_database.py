import pytest
from unittest.mock import patch, MagicMock
import psycopg2
from utils.database import retrieve_from_db

# --- FIXTURES ---

@pytest.fixture
def mock_user():
    """Creates a fake user object so we don't have to generate real ECC keys"""
    user = MagicMock()
    user.username = "TestUser"
    # Mocking the deep method chain: public_key.public_bytes().hex()
    user.public_key.public_bytes.return_value.hex.return_value = "fake_public_key_hex"
    return user

@pytest.fixture
def mock_signed_data():
    """Provides fake dictionary data formatted exactly like the real pipeline"""
    return {
        "sha256_hash": "fake_hash",
        "signature_hex": "fake_signature",
        "serialized_data": '{"author": "Test", "data": "Secret"}'
    }


# --- TESTS FOR RETRIEVE_FROM_DB ---

@patch("utils.database.psycopg2.connect")
def test_retrieve_from_db_success(mock_connect):
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cur
    
    # Simulate a successful database fetch 
    # (json_string, signature_hex, public_key_hex, username)
    mock_cur.fetchone.return_value = ('{"data": "secret"}', "sig123", "pub456", "TestUser")

    result = retrieve_from_db("meta-456")

    assert type(result) is dict
    assert result["json_string"] == '{"data": "secret"}'
    assert result["public_key_hex"] == "pub456"
    assert result["signature_hex"] == "sig123"

@patch("utils.database.psycopg2.connect")
def test_retrieve_from_db_not_found(mock_connect):
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cur
    
    # Simulate the database finding absolutely nothing
    mock_cur.fetchone.return_value = None

    result = retrieve_from_db("missing-uuid-999")

    assert result is None