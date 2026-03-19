import pytest
from unittest.mock import patch, MagicMock
import psycopg2
from data.database import save_to_db, retrieve_from_db

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

# --- TESTS FOR SAVE_TO_DB ---

@patch("psycopg2.connect")
def test_save_to_db_success(mock_connect, mock_user, mock_signed_data):
    # 1. Setup the fake database connection and cursor
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cur
    
    # 2. Tell the fake cursor exactly what to return when fetchone() is called.
    # First call returns user_uuid, second call returns metadata_uuid
    mock_cur.fetchone.side_effect = [("user-123",), ("meta-456",)]

    # 3. Run the function
    result = save_to_db(mock_user, mock_signed_data)

    # 4. Verify the function committed the save and returned the right UUID
    mock_conn.commit.assert_called_once()
    assert result == "meta-456"

@patch("psycopg2.connect")
def test_save_to_db_integrity_error(mock_connect, mock_user, mock_signed_data):
    # Setup mocks
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cur
    
    # Force the database to throw an IntegrityError (like a duplicate entry failure)
    mock_cur.execute.side_effect = psycopg2.IntegrityError("Simulated DB conflict")

    result = save_to_db(mock_user, mock_signed_data)

    # Verify the database rolled back to prevent corruption, and returned None
    mock_conn.rollback.assert_called_once()
    assert result is None

# --- TESTS FOR RETRIEVE_FROM_DB ---

@patch("psycopg2.connect")
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

@patch("psycopg2.connect")
def test_retrieve_from_db_not_found(mock_connect):
    mock_conn = MagicMock()
    mock_cur = MagicMock()
    mock_connect.return_value = mock_conn
    mock_conn.cursor.return_value = mock_cur
    
    # Simulate the database finding absolutely nothing
    mock_cur.fetchone.return_value = None

    result = retrieve_from_db("missing-uuid-999")

    assert result is None