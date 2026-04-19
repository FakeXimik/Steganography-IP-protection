# database.py
import os
import psycopg2
from dotenv import load_dotenv
import hashlib

load_dotenv()

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=5432,
        dbname=os.getenv("DB_NAME", "stegdb"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", "postgres")
    )

def register_user_db(username: str, public_key_hex: str) -> bool:
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        user_sql = """
            INSERT INTO users (username, public_key_hex)
            VALUES(%s, %s)
            ON CONFLICT (username) DO NOTHING
            RETURNING id
            """
        cur.execute(user_sql, (username, public_key_hex))
        
        # If no ID was returned, it means the user already existed!
        result = cur.fetchone()
        if not result:
            return False
            
        conn.commit()
        return True
    except Exception as e:
        print(f"--- DB ERROR: {e} ---")
        if conn: conn.rollback()
        return False
    finally:
        if conn:
            cur.close()
            conn.close()

def get_user_public_key(username: str):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT public_key_hex FROM users WHERE username = %s", (username,))
        res = cur.fetchone()
        return res[0] if res else None
    except Exception as e:
        print(f"--- ERROR: {e} ---")
        return None
    finally:
        if conn:
            cur.close()
            conn.close()

def get_username_by_public_key(public_key_hex: str):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT username FROM users WHERE public_key_hex = %s", (public_key_hex,))
        res = cur.fetchone()
        return res[0] if res else None
    finally:
        if conn:
            cur.close()
            conn.close()

def save_metadata_to_db(username: str, signature_hex: str, json_string: str):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute("SELECT id FROM users WHERE username = %s", (username,))
        res = cur.fetchone()
        if not res:
            return None
        user_uuid = res[0]

        sha256_hash = hashlib.sha256(json_string.encode('utf-8')).hexdigest()

        metadata_sql = """
            INSERT INTO metadata (user_id, sha256_hash, signature_hex, metadata_content)
            VALUES(%s, %s, %s, %s)
            RETURNING id;
            """
        cur.execute(metadata_sql, (user_uuid, sha256_hash, signature_hex, json_string))
        metadata_uuid = cur.fetchone()[0]

        conn.commit()
        return metadata_uuid
    except Exception as e:
        print(f"--- ERROR: {e} ---")
        if conn: conn.rollback()
        return None
    finally:
        if conn:
            cur.close()
            conn.close()

def retrieve_from_db(metadata_uuid: str):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        fetch_sql = """
            SELECT
                m.metadata_content::text,
                m.signature_hex,
                u.public_key_hex,
                u.username
            FROM 
                metadata m
            JOIN users u ON m.user_id = u.id
            WHERE m.id = %s
            """
        cur.execute(fetch_sql, (metadata_uuid,))
        result = cur.fetchone()
        
        if result:
            json_string, signature_hex, public_key_hex, username = result
            return {
                "json_string": json_string,
                "public_key_hex": public_key_hex,
                "signature_hex": signature_hex,
                "username": username
            }
        return None
    except Exception as e:
        print(f"--- DB READ ERROR: {e} ---")
        return None
    finally:
        if conn:
            cur.close()
            conn.close()