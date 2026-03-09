from models import *
from crypto import *
from database import *

#----------- basic tests

if __name__ == "__main__":
    print ('---------------------------------------------------')
    user1 = User("Cool dude")
    print(user1)
    print ('---------------------------------------------------')
    userData1 = UserMetadata("Sam", "This data is protected because I said so.").get_metadata()
    print(userData1)
    print ('---------------------------------------------------')

    signed_metadata = proccess_and_sign_metadata(userData1, user1.private_key)

    print(signed_metadata)
    print ('---------------------------------------------------')
    is_valid = verify_signature(signed_metadata['signature_hex'], user1.public_key, signed_metadata['serialized_data'])
    print ('---------------------------------------------------')
    if (is_valid):
        retrieve_data(signed_metadata['serialized_data'])
    print ('---------------------------------------------------')

    save_to_db(user1, signed_metadata)