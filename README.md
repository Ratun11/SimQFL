# SimQFL: Quantum Federated Learning Simulator

SimQFL is a web-based simulator for experimenting with **Federated Learning (FL)** and **Quantum Federated Learning (QFL)**. The project provides a Flask interface where users can configure clients, local epochs, global communication rounds, quantum circuit depth, and dataset choices, then observe simulation results in real time.

The simulator supports both classical FL using a CNN model and QFL using a PyTorch-based quantum neural network with a custom state-vector quantum circuit simulator. It also includes a CSV-upload workflow for running quantum federated regression on user-provided tabular data.

---

## Key Features

- **Web-based simulation interface** built with Flask, HTML, CSS, and JavaScript.
- **Classical Federated Learning** using a simple CNN model.
- **Quantum Federated Learning** using custom QNN models implemented with PyTorch.
- **Custom state-vector quantum circuit simulator** supporting single-qubit gates, rotation gates, controlled gates, and probability measurement.
- **Standard dataset support** for image-based FL/QFL experiments.
- **Custom CSV upload support** for tabular quantum federated regression.
- **Real-time training updates** using server-sent events.
- **Visualization of loss, accuracy, and MSE** using Chart.js.
- **Downloadable outputs**, including CSV results and graph exports as PNG/PDF.

---

## Project Structure

```text
SimQFL/
├── app.py
├── quantum_circuit_simulator.py
├── requirements.txt
├── temp_uploaded_data.csv
├── data/
│   ├── MNIST/
│   └── cifar-10-batches-py/
├── static/
│   ├── styles.css
│   └── images/
│       ├── qfl_overview.png
│       ├── qfl_overview.pdf
│       └── quantum_hero.jpg
└── templates/
    ├── index.html
    ├── about.html
    ├── contact.html
    ├── choose.html
    ├── simulate.html
    ├── simulate_own.html
    ├── simulation_results.html
    ├── simulation_results_custom.html
    ├── simulation_results_regression.html
    └── error.html
```

---

## Main Components

### 1. `app.py`

This is the main Flask application. It defines the web routes, model classes, dataset loading, federated training loops, streaming responses, and result downloads.

Important components include:

- `SimpleCNN`: classical CNN model for standard FL.
- `QNN_MNIST`: quantum neural network for MNIST and FashionMNIST-style inputs.
- `QNN_CIFAR`: quantum neural network for CIFAR-style image inputs.
- `QNN_Regressor`: quantum neural network for tabular regression.
- `partition_dataset()`: divides data among multiple clients.
- `average_weights()`: performs federated averaging of local model parameters.
- `pad_features()`: pads tabular features to match the required quantum state dimension.
- `/stream_simulation`: streams classification simulation results.
- `/stream_simulation_regression`: streams regression simulation results.
- `/download_csv`: downloads simulation results as a CSV file.

### 2. `quantum_circuit_simulator.py`

This file contains the custom quantum circuit simulator. It uses PyTorch tensors to represent quantum state vectors and apply quantum gates.

Supported operations include:

- Pauli gates: `X`, `Y`, `Z`
- Hadamard gate: `H`
- Rotation gates: `Rx`, `Ry`, `Rz`
- General rotation gate: `R`
- Controlled gates: `CX`, `CZ`
- Layered rotation operations
- Linear entangling layers
- Measurement probability computation

### 3. `templates/`

This folder contains the HTML pages used by Flask.

Main pages include:

- Home page
- About page with QFL explanation
- Contact page
- Simulation configuration page
- Custom CSV upload page
- Simulation result pages
- Error page

### 4. `static/`

This folder contains CSS styling and image assets used in the web interface.

---

## Technology Stack

- Python
- Flask
- PyTorch
- TorchVision
- NumPy
- Pandas
- HTML/CSS/JavaScript
- Chart.js
- jsPDF

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ratun11/SimQFL.git
cd SimQFL
```

### Step 2: Create a Virtual Environment

On Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Application

Start the Flask app with:

```bash
python app.py
```

Then open your browser and go to:

```text
http://127.0.0.1:5000
```

---

## How to Use the Simulator

### Standard Dataset Simulation

1. Open the home page.
2. Go to the simulation configuration page.
3. Select the learning type:
   - Federated Learning
   - Quantum Federated Learning
4. Select a dataset.
5. Set the number of clients, local epochs, global epochs, qubits, and quantum layers.
6. Start the simulation.
7. View real-time loss and accuracy updates.
8. Download results as CSV, PNG, or PDF.

### Custom CSV Regression Simulation

1. Go to the custom data simulation page.
2. Upload a CSV file.
3. Select input features.
4. Select the output feature to predict.
5. Start the quantum federated regression simulation.
6. View MSE values across global epochs.
7. Download the MSE chart or result CSV.

---

## Supported Simulation Modes

| Mode | Description | Output Metric |
|---|---|---|
| Classical FL | Federated learning with a CNN model | Loss, Accuracy |
| Quantum FL | Federated learning with a QNN model | Loss, Accuracy |
| Quantum FL Regression | QNN-based federated regression on CSV data | Mean Squared Error |

---

## Quantum Federated Learning Workflow

The QFL workflow follows these steps:

1. The global QNN model is initialized on the server.
2. The training dataset is partitioned among multiple clients.
3. Each client receives the current global model.
4. Clients train locally using their private data.
5. Local model parameters are sent back to the server.
6. The server aggregates local parameters using federated averaging.
7. The updated global model is evaluated.
8. The process repeats for the selected number of global epochs.

---

## Quantum Model Design

The QNN models use amplitude-style input normalization and a parameterized quantum circuit. Each quantum layer applies trainable `Ry` rotations followed by a linear CNOT entangling layer. The final quantum state probabilities are passed into a classical linear layer for prediction.

For a quantum circuit with `n` qubits, the input dimension must match `2^n`. For custom tabular data, feature vectors are padded with zeros when necessary.

---

## Output and Visualization

The simulator displays training results in real time. Depending on the simulation type, it shows:

- Global epoch
- Test loss
- Test accuracy
- Mean squared error

The interface also provides options to download:

- Simulation results as CSV
- Loss graph as PNG/PDF
- Accuracy graph as PNG/PDF
- MSE graph as PNG/PDF

---

## Dataset Notes

The code can download standard datasets through TorchVision when needed. To keep the GitHub repository lightweight, large downloaded dataset files should usually not be committed to the repository.

Recommended `.gitignore` entries:

```gitignore
__pycache__/
*.pyc
venv/
.venv/
.env
.DS_Store
*.zip
*.tar.gz
data/
temp_uploaded_data.csv
```

If datasets are required for a specific experiment, describe how to download them instead of pushing large raw dataset files to GitHub.

---

## Important Reproducibility Notes

- The simulator uses PyTorch and automatically selects CUDA if a GPU is available.
- CPU execution is supported, but QNN simulations can be slower for larger qubit counts.
- Increasing the number of qubits increases the state-vector dimension as `2^n`.
- Large numbers of clients, local epochs, or global epochs can significantly increase runtime.
- For fast testing, start with a small number of clients and epochs.

Example quick test configuration:

```text
Clients: 3
Local epochs: 1
Global epochs: 5
Qubits: 10
Quantum layers: 1
```

---

## Repository Hygiene Before Pushing to GitHub

Before committing, avoid tracking large generated files, temporary files, or dataset archives. GitHub blocks files larger than 100 MB, so large dataset archives should be removed from the commit history or excluded before pushing.

Useful commands:

```bash
git status
git add README.md .gitignore app.py quantum_circuit_simulator.py requirements.txt static templates
git commit -m "Add SimQFL project code and documentation"
git push origin main
```

---

## Contact

For questions or collaboration, please contact:

**Ratun Rahman**  
The University of Alabama in Huntsville  
Email: `rr0110@uah.edu`

---

## License

No license file is currently included in this repository. Add a license file before public release if you want others to use, modify, or distribute the project.
