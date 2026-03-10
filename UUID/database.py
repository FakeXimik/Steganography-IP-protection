import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def save_to_db(user_obj, signed_data: dict):

    try:
        conn = psycopg2.connect(
            host = 'localhost',
            port = 5432,
            dbname = os.getenv("DB_NAME"),
            user = os.getenv("DB_USER"),
            password = os.getenv("DB_PASSWORD")
            )
        
        cur = conn.cursor()

  #ON CONFLICT prevents Postgres from giving the error when updating existing user
        #We update the user, not changing anything and get the UUID
        user_sql = """
            INSERT INTO users (username, public_key_hex)
            VALUES(%s, %s)
            ON CONFLICT (username) DO UPDATE  
            SET username = EXCLUDED.username
            RETURNING id
            """
      
        #Serializing public key to hex 
        from cryptography.hazmat.primitives import serialization
        public_hex = user_obj.public_key.public_bytes(
            encoding = serialization.Encoding.DER,
            format = serialization.PublicFormat.SubjectPublicKeyInfo
        ).hex()

        cur.execute(user_sql, (user_obj.username, public_hex))
        user_uuid = cur.fetchone()[0]

        metadata_sql = """
            INSERT INTO metadata (user_id, sha256_hash, signature_hex, metadata_content)
            VALUES(%s, %s, %s, %s)
            RETURNING id;
            """
        
        cur.execute(metadata_sql,(
                    user_uuid,
                    signed_data["sha256_hash"],
                    signed_data["signature_hex"],
                    signed_data["serialized_data"]) # Converts to jsonB
        )

        metadata_uuid = cur.fetchone()[0]

        conn.commit()
        print(f"Metadata saved to DB with UUID: {metadata_uuid}")
        return metadata_uuid

    except psycopg2.IntegrityError as e:
        conn.rollback()
        print(f"--- DB ERROR: {e} ---")

    except Exception as e:
        print(f"--- ERROR: {e} ---")

    finally:
        if conn:
            cur.close()
            conn.close()

# -------------------------------------

def retrieve_from_db(metadata_uuid: str):
    conn = None

    try:
        conn = psycopg2.connect(
        host = 'localhost',
        port = 5432,
        dbname = os.getenv("DB_NAME"),
        user = os.getenv("DB_USER"),
        password = os.getenv("DB_PASSWORD")
        )

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
            print(f"Successfully fetched file {metadata_uuid}, (Signed by: {username})")

            return {
                "json_string": json_string,
                "public_key_hex": public_key_hex,
                "signature_hex": signature_hex
            }
        else:
            print(f"No records of {metadata_uuid} found")
            return None
        
    except Exception as e:
        print(f"--- DB READ ERROR: {e} ---")

    finally:
        if conn:
            cur.close()
            conn.close()
        
    
        
        