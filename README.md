# AegisDim

AegisDim is an adaptive multimodal Matryoshka feature compression framework guided by topological dimensions. It utilizes Local Intrinsic Dimensionality (LID) as a theoretical lower bound to strictly constrain the dynamic dimension truncation process of neural networks. This approach significantly saves retrieval computational costs while preventing semantic space collapse under out of distribution scenarios at the fundamental physics level.

### Environment Setup

To maintain a clean system environment and avoid dependency conflicts, please make sure to use a Python virtual environment.

1. Create a virtual environment named 'venv'
python3 -m venv venv

2. Activate the virtual environment
source venv/bin/activate

3. Install core dependencies
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.11 python3.11-venv -y
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft llm2vec datasets

### Directory Structure

* data/ Stores raw and processed evaluation datasets
* models/ Stores core network components
  * backbone.py Responsible for loading the strictly frozen foundational multimodal large model
  * smec_adaptor.py Contains the residual feature adaptor and the scoring-based dynamic truncation module
* scripts/ Stores execution scripts and testing scaffolds
  * test_smec_forward.py A minimal testing script to verify the forward propagation of tensors
* train_smec.py The main training loop for joint optimization

### Quick Verification

Run the forward propagation testing script in the root directory to ensure the tensor dimension-slicing logic of the adaptive module strictly matches the physical expectations:

python scripts/test_smec_forward.py