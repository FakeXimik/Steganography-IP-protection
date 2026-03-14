import reedsolo
import uuid
import random
from typing import Tuple

class RSCodecPipeline:
    """
    Forward Error Correction (FEC) pipeline using Reed-Solomon.
    Bridges the gap between the probabilistic neural extraction network 
    and the deterministic database pointer.
    """
    
    def __init__(self, parity_symbols: int = 10):
        """
        Initializes the Reed-Solomon Codec.
        'parity_symbols' dictates how much mathematical padding we add.
        10 parity bytes means we can correct up to 5 completely corrupted bytes.
        """
        self.parity_symbols = parity_symbols
        self.rs = reedsolo.RSCodec(self.parity_symbols)

    def encode_uuid(self, raw_uuid_string: str) -> bytes:
        """
        Ingests a raw UUID string, applies Reed-Solomon encoding, 
        and outputs the padded binary payload.
        """
        # 1. The Input: Convert the standard UUID string into raw bytes.
        # A standard 128-bit UUID perfectly translates into 16 bytes of data.
        uuid_obj = uuid.UUID(raw_uuid_string)
        uuid_bytes = uuid_obj.bytes
        
        # 2. Apply the encoding to generate the parity bytes and append them.
        encoded_payload = self.rs.encode(uuid_bytes)
        
        # 3. The Output: Return the final, padded binary payload ready for the neural network.
        return encoded_payload

    def decode_payload(self, noisy_payload: bytes) -> Tuple[bool, str]:
        """
        Takes the noisy payload from the neural network, runs the reedsolo decoding algorithm 
        to strip away parity bytes and fix any burst errors, and outputs the clean UUID string.
        
        Returns:
            Tuple[bool, str]: A success flag and either the recovered UUID string or an error message.
        """
        try:
            # decode() returns a tuple: (decoded_message, decoded_message_plus_ecc, errATA)
            # We only need the first part (the original message bytes).
            decoded_bytes = self.rs.decode(noisy_payload)[0]
            
            # Convert the 16 bytes back into a standard UUID string
            clean_uuid = str(uuid.UUID(bytes=bytes(decoded_bytes)))
            return True, clean_uuid
            
        except reedsolo.ReedSolomonError as e:
            # Graceful Error Handling: Return False instead of crashing the API
            return False, f"Unrecoverable error: {e}"
        except Exception as e:
            return False, f"Unexpected decoding error: {e}"

    @staticmethod
    def simulate_burst_error(payload: bytes, start_idx: int, num_bytes: int) -> bytes:
        """
        Simulates an image crop or continuous localized distortion (burst error) 
        by replacing a sequence of bytes with zeros.
        """
        corrupted = bytearray(payload)
        end_idx = min(start_idx + num_bytes, len(corrupted))
        for i in range(start_idx, end_idx):
            corrupted[i] = 0
        return bytes(corrupted)

# --- Let's test it immediately! ---
if __name__ == "__main__":
    print("\n" + "="*50)
    print(" FEC / REED-SOLOMON OOP PIPELINE TEST")
    print("="*50)
    
    # 1. Generate a fake UUID that Student A might hand to you
    sample_uuid = str(uuid.uuid4())
    print(f"[1] Original UUID   : {sample_uuid}")
    
    # Initialize our new Pipeline class
    parity_symbols = 10
    fec_pipeline = RSCodecPipeline(parity_symbols=parity_symbols)
    
    # 2. Encode
    protected_payload = fec_pipeline.encode_uuid(sample_uuid)
    print(f"[2] Padded Payload  : {list(protected_payload)}")
    print(f"    Payload Length  : {len(protected_payload)} bytes (16 raw + {parity_symbols} parity)")
    
    # 3. Verify the Threshold: Test exact mathematical limits
    # Reed-Solomon can fix floor(parity_symbols / 2) completely corrupted bytes (burst errors)
    max_threshold = parity_symbols // 2
    total_bits = len(protected_payload) * 8
    
    print(f"\n[3] Simulating Burst Errors (Threshold = Correct up to {max_threshold} bytes)")
    
    # We will test burst sizes from 0 up to threshold + 1 to prove it repairs 
    # everything under the threshold and fails right when it should.
    for burst_size in range(max_threshold + 2):
        print(f"\n--- Testing Burst Size: {burst_size} bytes ---")
        
        # Corrupt contiguous bytes in the middle of our payload
        start_idx = 4
        noisy_payload = RSCodecPipeline.simulate_burst_error(protected_payload, start_idx, burst_size)
        
        # Calculate exact Bit Error Rate (BER)
        bit_flips = 0
        for i in range(len(protected_payload)):
            diff = protected_payload[i] ^ noisy_payload[i]
            bit_flips += bin(diff).count("1")
        ber = bit_flips / total_bits
        
        print(f"    Corrupted Data  : {list(noisy_payload)}")
        print(f"    Bit Error Rate  : {ber:.2%} ({bit_flips}/{total_bits} bits damaged)")
        
        # 4. Decode using Graceful Error Handling
        success, result = fec_pipeline.decode_payload(noisy_payload)
        
        if success:
            # Perfect retrieval check
            assert result == sample_uuid, "Mismatch: Recovered UUID != Original!"
            print(f"    Status          : \033[92m[SUCCESS] Perfect recovery!\033[0m UUID: {result}")
        else:
            print(f"    Status          : \033[91m[FAILED] {result}\033[0m")
            if burst_size <= max_threshold:
                print("    WARNING: Decoder failed below the threshold!")
            else:
                print(f"    (Expected failure: damage of {burst_size} bytes exceeds threshold of {max_threshold} bytes)")

    print("\n" + "="*50)