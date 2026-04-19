# Steganography IP Protection 

[![Python >=3.11](https://img.shields.io/badge/python->=3.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.10.0](https://img.shields.io/badge/PyTorch-2.10.0-EE4C2C.svg)](https://pytorch.org/)
[![CI](https://github.com/FakeXimik/Steganography-IP-protection/actions/workflows/python-app.yml/badge.svg?branch=main)](https://github.com/FakeXimik/Steganography-IP-protection/actions/workflows/python-app.yml)


An advanced, machine-learning-based steganography system designed to protect digital image copyright. This project combines **Generative Adversarial Networks (GANs)**, **Forward Error Correction (Reed-Solomon)**, and **Elliptic Curve Cryptography (ECC)** to invisibly embed and extract cryptographic ownership proof from images.

---

## Architecture Overview: The "License Plate" Model

Embedding heavy JSON metadata and 72-byte cryptographic signatures directly into an image destroys pixel fidelity. Instead, this system uses a "License Plate" architecture:

1. **Cryptographic Signing (ECC):** Author metadata is serialized to JSON, hashed via SHA-256, and signed using an Elliptic Curve private key.
2. **Database Storage:** The heavy metadata and ECC signature are securely stored in a PostgreSQL database and assigned a unique 16-byte `UUID`.
3. **Forward Error Correction (FEC):** The 16-byte UUID is passed through a Reed-Solomon pipeline (`RSCodecPipeline`), adding 10 bytes of parity data. This creates a highly robust **208-bit payload** capable of surviving image distortion and burst errors.
4. **Adversarial Steganography (HiDDeN):** 
   * The **Encoder** hides the 208-bit payload inside the pixels of a cover image.
   * A `StandardNoiseLayer` simulates real-world compression and distortion.
   * The **Discriminator** attempts to catch the alterations, forcing the Encoder to make the hidden data mathematically invisible.
5. **Extraction:** A **Decoder** network reads the noisy pixels and extracts the bits. The FEC pipeline repairs any damage to reconstruct the original UUID, which is then used to fetch the cryptographic proof from the database.

---

## Project Structure

```text
Steganography-IP-protection/
├── data/               # Custom DataLoaders (CIFAR-100 with normalization)
├── models/             # Neural Network layers (Encoder, Decoder, Discriminator, Noise)
├── utils/              # Cryptography, PostgreSQL connections, and Reed-Solomon FEC
├── tests/              # Pytest suite with mocked datasets for CI/CD
├── .github/workflows/  # GitHub Actions CI pipeline configuration
├── static/             # Static web assets and styles
├── extract.html        # Front-end payload extraction interface
├── hide.html           # Front-end payload embedding interface
├── main.py             # CLI pipeline demonstrating LSB Steganography and ECC processes
├── api.py              # FastAPI server orchestrating the web interface and backend
├── Dockerfile          # Configuration for containerization of the web app
├── docker-compose.yml  # Orchestrates the API container alongside a PostgreSQL database
├── pyproject.toml      # Pinned dependencies (strict reproducibility)
├── train.py            # Master GPU training loop
└── README.md
```

---

## Installation & Setup

**Prerequisites:** Python 3.11 or higher.

1. **Clone the repository and set up a virtual environment:**
   ```bash
   git clone https://github.com/YourUsername/Steganography-IP-protection.git
   cd Steganography-IP-protection
   python -m venv .venv
   ```
   
2. **Activate the environment:**
  
    Windows: `.venv\Scripts\activate`

    Mac/Linux: `source .venv/bin/activate`

3. **Install Core Project & Testing Dependencies:** 
  This command installs the exact package versions specified in pyproject.toml, including pytest for local testing.
    ```bash
    python -m pip install --upgrade pip
    pip install -e .[test]
    ```

4. **Hardware Acceleration (NVIDIA RTX 4000 series / CUDA 12.8):** 
  By default, the CI pipeline and base install use the CPU version of PyTorch. To unlock your local GPU for training, override it with the pinned CUDA version:

    ```bash
    pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
    ```

5. **Verify your GPU is active:**
    ```bash
    python -c "import torch; print(torch.cuda.is_available())"
    ```

---

## Deployment with Docker

This project includes fully configured Docker components to serve the FastAPI application and deploy the necessary PostgreSQL database securely. The API container also supports GPU reservation via the NVIDIA runtime.

**Prerequisites:** Docker, Docker Compose, and NVIDIA Container Toolkit (if utilizing GPU).

1. **Configure Environment Variables:**
   Create a `.env` file in the root directory and ensure the database credentials are set:
   ```env
   DB_USER=your_user
   DB_PASSWORD=your_password
   DB_NAME=your_database_name
   ```

2. **Build and Spin Up the Containers:**
   Launch the system using Docker Compose. This starts both the `stego_app_prod` API server and the `IP_postgres` database container.
   ```bash
   docker-compose up --build -d
   ```

3. **Database Initialization:**
   The `init.sql` script is automatically mounted to Docker and initializes the necessary ledger table schemas on first boot. Persistent SQL data is seamlessly mapped to the `./postgres_data` volume.

---

## Running the Components

### 1. Web Application via FastAPI
If you prefer running the API locally outside of Docker, start the FastAPI web server from your virtual environment:
```bash
fastapi run api.py
```
This exposes the core web application endpoints locally at `http://127.0.0.1:8000`:
- **/** : The `hide.html` interface to encrypt, compress, and embed data.
- **/extract** : The `extract.html` interface to upload a suspected image, recover the UUID, and retrieve database proof.
- **/about** : Informational overview page (`about.html`).

### 2. Basic Command Line Pipeline (Demo)
Run the core operational pipeline without launching the interface. This script (`main.py`) acts as a demonstration, stepping through user generation, key creation, database transactions, bit-insertion (using an LSB fallback method), and final extraction logic.
```bash
python main.py
```

---

## Running Tests (CI/CD)

This project is protected by a continuous integration pipeline via GitHub Actions. The test suite uses pytest-mock to simulate large datasets (like CIFAR-100), ensuring rapid execution without downloading gigabytes of data.

To run the test suite locally before committing code:
```bash
pytest -v
```

---

## Training the Model

The main training loop (`train.py`) orchestrates the Generator vs. Discriminator battle. It dynamically detects your CUDA device and ingests real Reed-Solomon UUIDs to train the network on production-accurate data.

```bash
python train.py
```

### Understanding Training Metrics
While the model trains, watch your terminal for these key indicators:

**Total Loss / D_Loss:** The Generator and Discriminator fighting. These should fluctuate and trend downwards.

**BER (Bit Error Rate):** The most critical metric. An untrained model starts near 50.0% (random guessing). As the Decoder learns to read the embedded UUIDs through the noise layer, this percentage will drop.

Note: Hiding 208 bits inside a tiny 32x32 CIFAR-100 image is a dense steganographic task. If the BER plateaus near 50%, increase the epochs (e.g., 50-100) or swap the dataloader to use larger cover images (128x128).
