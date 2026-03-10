from cryptography.hazmat.primitives import serialization
import json
from UUID.models import *
from UUID.crypto import *
from UUID.database import *
from ReedSolomon.fec_pipeline import *
from LSBSteg.LSBSteg import *


#--- Pipeline test

if __name__ == "__main__":

    # Generate user + dummy metadata and sign it
    print ('\n--- CREATE USER ANR PERFORM OPERATIONS WITH DB ---')
    user1 = User("Cool dude", b"Cool password")
    userData1 = UserMetadata("Sam", "This data is protected because I said so.").get_metadata()
    print(user1)
    print(userData1)
    print ('\n--- SIGNING METADATA ---')
    signed_metadata = proccess_and_sign_metadata(userData1, user1.private_key)
    print(signed_metadata)

    # Save to db and get the UUID
    print ('\n--- Retrieve the UUID from the base ---')
    saved_uuid = save_to_db(user1, signed_metadata)
    
    if saved_uuid:
        print("SIGNED METADATA HAS BEEN SAVED TO THE DB")
        print(f"Item's UUID is: {saved_uuid}")

    print ('\n--- REED-SOLOMON ENCODING & SIMULATION ---')

    fec = RSCodecPipeline(parity_symbols = 10) #Init

    payload_bytes = fec.encode_uuid(str(saved_uuid)) #Encode
    print(f"Protected payload ready to hide in image: {list(payload_bytes)}")
    
    #Damage the payload
    damaged_payload = fec.simulate_burst_error(payload_bytes, start_idx=2, num_bytes=4)
    print(print(f"!!! Damaged payload pulled from image: {list(damaged_payload)}"))
    
    success, recovered_uuid = fec.decode_payload(damaged_payload)

    if success:
        print(f"Fixed the damage! Clean UUID: {recovered_uuid}")
    else:
        print("The damage was too heavy to fix.")
        recovered_uuid = None # Fails the next step

    print ('---------------------------------------------------')

    db_data = retrieve_from_db(recovered_uuid)


    if db_data:
        print("\n --- SIGNED METADATA HAS BEEN RETRIEVED FROM THE DB ---")

        # Transform from public key from hex and reconstruct json string
        public_key_bytes = bytes.fromhex(db_data["public_key_hex"])
        restored_public_key = serialization.load_der_public_key(public_key_bytes)

        recovered_dict = json.loads(db_data['json_string'])
        
        # Re-stringify it using Python's rules 
        perfect_json_string = json.dumps(recovered_dict, sort_keys=True)

        print ('\n --- VERIFYING THE INTEGRITY ---')
        # Verify the integrity
        is_valid = verify_signature(db_data['signature_hex'], restored_public_key, perfect_json_string)
        print(db_data['signature_hex'])
        print ('---------------------------------------------------')
        if (is_valid):
            retrieve_data(db_data['json_string'])
            
            print ('---------------------------------------------------')
