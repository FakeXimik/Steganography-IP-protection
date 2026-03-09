from models import *
from crypto import *
from database import *
from cryptography.hazmat.primitives import serialization

#----------- basic tests

if __name__ == "__main__":
    # Generate user + dummy metadata and sign it
    print ('---------------------------------------------------')
    user1 = User("Cool dude 2")
    print(user1)
    print ('---------------------------------------------------')
    userData1 = UserMetadata("Sam", "This data is protected because I said so.").get_metadata()
    print(userData1)
    print ('---------------------------------------------------')
    signed_metadata = proccess_and_sign_metadata(userData1, user1.private_key)
    print(signed_metadata)

# Save and retrieve from db
    print ('---------------------------------------------------')
    saved_uuid = save_to_db(user1, signed_metadata)
    
    if saved_uuid:
        print("SIGNED METADATA HAS BEEN SAVED TO THE DB")

    print ('---------------------------------------------------')
    db_data = retrieve_from_db(saved_uuid)


    if db_data:
        print("SIGNED METADATA HAS BEEN RETRIEVED FROM THE DB")

        # Transform from public key from hex and reconstruct json string
        public_key_bytes = bytes.fromhex(db_data["public_key_hex"])
        restored_public_key = serialization.load_der_public_key(public_key_bytes)

        import json
        recovered_dict = json.loads(db_data['json_string'])
        
        # Re-stringify it using Python's rules 
        perfect_json_string = json.dumps(recovered_dict, sort_keys=True)

        print ('---------------------------------------------------')
        # Verify the integrity
        is_valid = verify_signature(db_data['signature_hex'], restored_public_key, perfect_json_string)
        print(db_data['signature_hex'])
        print ('---------------------------------------------------')
        if (is_valid):
            retrieve_data(db_data['json_string'])
            
            print ('---------------------------------------------------')
