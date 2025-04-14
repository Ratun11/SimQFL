SimQFL: Quantum Federated Learning Simulator
============================================

SimQFL is a Flask-based web simulator that allows users to explore both classical and quantum federated learning models using datasets like MNIST, FashionMNIST, CIFAR-10, and CIFAR-100. It also supports regression on user-uploaded CSV data with quantum backends.

----------------------------------------
Requirements
----------------------------------------

- Python 3.7+
- pip (Python package manager)

Required Libraries:

Run the following command to install required libraries:

    pip install -r requirements.txt

If you don’t have a virtual environment set up, it is recommended to do so:

    python -m venv venv
    source venv/bin/activate       (Linux/Mac)
    venv\Scripts\activate          (Windows)

----------------------------------------
Running the Application
----------------------------------------

1. Open terminal or command prompt.
2. Navigate to the project directory.
3. Run:

    python app.py

4. Visit http://127.0.0.1:5000 in your browser to use the simulator.

----------------------------------------
Folder Structure
----------------------------------------

SimQFL/
│
├── app.py                        # Main Flask application
├── quantum_circuit_simulator.py # Quantum logic and utilities
├── requirements.txt             # Python dependencies
├── templates/                   # HTML templates
├── static/                      # JavaScript, CSS, assets
├── MNIST/, CIFAR10/, etc.       # Dataset folders (used by torchvision)
└── venv/                        # Virtual environment (optional)

----------------------------------------
Key Features
----------------------------------------

- Supports Classical and Quantum Federated Learning
- Customizable quantum settings: qubits, layers, encoding
- Built-in support for MNIST, CIFAR, FashionMNIST
- Custom CSV upload for quantum regression tasks
- Real-time progress streaming with results
- Download final simulation results as CSV

----------------------------------------
Deployment on Render
----------------------------------------

1. Push this project to a GitHub repository.
2. On https://render.com:
    - New Web Service → Connect to your GitHub
    - Set:
        Build Command: pip install -r requirements.txt
        Start Command: gunicorn app:app

----------------------------------------
License
----------------------------------------

This project is released under the MIT License.
