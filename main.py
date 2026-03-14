from cryptography.hazmat.primitives import serialization
import json
import cv2
from models.user import *
from models.userMetadata import *
from utils.crypto import *
from data.database import *
from utils.fec import *
from models.LSBSteg import *

if __name__ == "__main__":

    # Generate user + dummy metadata and sign it
    print ('\n--- CREATE USER AND PERFORM OPERATIONS WITH DB ---')
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

    print ('\n--- REED-SOLOMON ENCODING ---')
    fec = RSCodecPipeline(parity_symbols = 10) 

    # Encode the UUID into bytes
    payload_bytes = fec.encode_uuid(str(saved_uuid)) 
    print(f"Protected payload ready to hide: {list(payload_bytes)}")
    
    print('\n--- LSB STEGANOGRAPHY (HIDING IN IMAGE) ---')

    
    # Load your cover image from the assets folder
    carrier_image = cv2.imread('assets/test_img.png')
    
    if carrier_image is None:
        print("Error: Could not find 'assets/test_img.png'. Please add it to the folder.")
        exit()

    # Hide the payload
    steg_hider = LSBSteg(carrier_image)
    encoded_image = steg_hider.encode_binary(payload_bytes)
    
    # Save the protected image inside the assets folder
    cv2.imwrite('assets/protected_image.png', encoded_image)
    print("Success: Payload hidden and saved to 'assets/protected_image.png'")

    print ('\n--- LSB STEGANOGRAPHY (EXTRACTING FROM IMAGE) ---')
    
    # Read the protected image back from the assets folder
    loaded_image = cv2.imread('assets/protected_image.png')
    
    # Extract the hidden bytes
    steg_extractor = LSBSteg(loaded_image)
    extracted_payload = steg_extractor.decode_binary()
    print(f"Raw payload pulled from image: {list(extracted_payload)}")

    print ('\n--- REED-SOLOMON DECODING ---')
    # Decode directly from the extracted payload
    success, recovered_uuid = fec.decode_payload(extracted_payload)

    if success:
        print(f"Clean UUID extracted and verified: {recovered_uuid}")
    else:
        print("Failed to decode the payload.")
        recovered_uuid = None

    print ('---------------------------------------------------')

    db_data = retrieve_from_db(recovered_uuid)

    if db_data:
        print("\n --- SIGNED METADATA HAS BEEN RETRIEVED FROM THE DB ---")

        public_key_bytes = bytes.fromhex(db_data["public_key_hex"])
        restored_public_key = serialization.load_der_public_key(public_key_bytes)

        recovered_dict = json.loads(db_data['json_string'])
        
        perfect_json_string = json.dumps(recovered_dict, sort_keys=True)

        print ('\n --- VERIFYING THE INTEGRITY ---')
        is_valid = verify_signature(db_data['signature_hex'], restored_public_key, perfect_json_string)
        print(f"Signature: {db_data['signature_hex']}")
        
        print ('---------------------------------------------------')
        if is_valid:
            retrieve_data(db_data['json_string'])
            print ('---------------------------------------------------')