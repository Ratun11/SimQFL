SimQFL: Quantum Federated Learning Simulator
============================================

SimQFL is a Flask-based web simulator that allows users to explore both classical and quantum federated learning models using datasets like MNIST, FashionMNIST, CIFAR-10, and CIFAR-100. It also supports regression on user-uploaded CSV data with quantum backends.

----------------------------------------
Downloads
----------------------------------------

You can download the files from here:
https://drive.google.com/drive/folders/1tjTHZ-n1wasdPprprj7oglfZSXT3lRYr?usp=sharing

----------------------------------------
Running the Application
----------------------------------------

1. Open the Google Drive link and download the files
2. Open the dist folder and the app folder
3. Run:

    app.exe

4. Visit http://127.0.0.1:5000 in your browser to use the simulator (if it doesn't open automically).

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
