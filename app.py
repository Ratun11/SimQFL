from flask import Flask, render_template, request, Response, jsonify, redirect, url_for
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import random
import copy
import json
import time
import pandas as pd
import os

# Import the quantum circuit simulator (ensure this module is available)
from quantum_circuit_simulator import quantum_circuit

app = Flask(__name__)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global variable to store simulation results for CSV download (for classification simulation)
last_simulation_results = []

# Global variable to store selected feature configuration for regression simulation.
# It will be a dictionary with two keys: "selected_features" and "output_feature".
selected_feature_config = {}

# -------------------------------
# Define a simple CNN model (for classical FL)
# -------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, input_channels, num_classes, image_size):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_dim = 64 * (image_size // 4) * (image_size // 4)
        self.fc1 = nn.Linear(self.flatten_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# -------------------------------
# Define QNN models for Quantum FL (Classification)
# -------------------------------
# QNN model for MNIST and FashionMNIST (with padding)
class QNN_MNIST(nn.Module):
    def __init__(self, n, L):
        super(QNN_MNIST, self).__init__()
        self.n = n
        self.L = L
        self.flatten = nn.Flatten()
        angles = torch.empty((L, n), dtype=torch.float64)
        torch.nn.init.uniform_(angles, -0.01, 0.01)
        self.angles = nn.Parameter(angles)
        self.linear = nn.Linear(2**n, 10)  # 10 classes

    def forward(self, x):
        # Use padding as in your original QNN for MNIST
        x = F.pad(x, (2, 2, 2, 2), "constant", 0)
        x = self.flatten(x)
        x /= torch.linalg.norm(x.clone(), ord=2, dim=1, keepdim=True)
        qc = quantum_circuit(num_qubits=self.n, state_vector=x.T)
        for l in range(self.L):
            qc.Ry_layer(self.angles[l].to(torch.cfloat))
            qc.cx_linear_layer()
        x = torch.real(qc.probabilities())
        x = self.linear(x.T)
        return x

# QNN model for CIFAR-10 and CIFAR-100 (without padding)
class QNN_CIFAR(nn.Module):
    def __init__(self, n, L, num_classes=10):
        super(QNN_CIFAR, self).__init__()
        self.n = n
        self.L = L
        self.flatten = nn.Flatten()
        angles = torch.empty((L, n), dtype=torch.float64)
        torch.nn.init.uniform_(angles, -0.01, 0.01)
        self.angles = nn.Parameter(angles)
        self.linear = nn.Linear(2**n, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x /= torch.linalg.norm(x.clone(), ord=2, dim=1, keepdim=True)
        if x.shape[1] != 2**self.n:
            raise ValueError(f"Input size {x.shape[1]} does not match 2^{self.n} = {2**self.n}.")
        qc = quantum_circuit(num_qubits=self.n, state_vector=x.T)
        for l in range(self.L):
            qc.Ry_layer(self.angles[l].to(torch.cfloat))
            qc.cx_linear_layer()
        x = torch.real(qc.probabilities())
        x = self.linear(x.T)
        return x

# -------------------------------
# New Model: QNN_Regressor for Regression using Tabular Data
# -------------------------------
class QNN_Regressor(nn.Module):
    def __init__(self, n, L):
        super(QNN_Regressor, self).__init__()
        self.n = n
        self.L = L
        self.flatten = nn.Flatten()
        angles = torch.empty((L, n), dtype=torch.float64)
        torch.nn.init.uniform_(angles, -0.01, 0.01)
        self.angles = nn.Parameter(angles)
        # Output single continuous value
        self.linear = nn.Linear(2**n, 1)

    def forward(self, x):
        x = self.flatten(x)
        # Normalize each sample (avoid division by zero if all zeros)
        norm = torch.linalg.norm(x.clone(), ord=2, dim=1, keepdim=True)
        norm[norm == 0] = 1.0
        x = x / norm
        qc = quantum_circuit(num_qubits=self.n, state_vector=x.T)
        for l in range(self.L):
            qc.Ry_layer(self.angles[l].to(torch.cfloat))
            qc.cx_linear_layer()
        x = torch.real(qc.probabilities())
        x = self.linear(x.T)
        return x

# -------------------------------
# Helper Functions (common)
# -------------------------------
def partition_dataset(dataset, num_clients):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    client_indices = np.array_split(indices, num_clients)
    return [list(arr) for arr in client_indices]

def average_weights(state_dicts):
    avg_state_dict = copy.deepcopy(state_dicts[0])
    for key in avg_state_dict.keys():
        for i in range(1, len(state_dicts)):
            avg_state_dict[key] += state_dicts[i][key]
        avg_state_dict[key] = avg_state_dict[key] / len(state_dicts)
    return avg_state_dict

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    preds = []
    targets = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            loss = criterion(output, y)
            total_loss += loss.item() * X.size(0)
            preds.append(output)
            targets.append(y)
            total_samples += X.size(0)
    avg_loss = total_loss / total_samples
    return avg_loss

# -------------------------------
# New helper for regression: pad features to dimension 2^n
# -------------------------------
def pad_features(X, n):
    # X is a numpy array with shape (num_samples, m)
    num_samples, m = X.shape
    target_dim = 2 ** n
    if m > target_dim:
        # Truncate if more than required
        return X[:, :target_dim]
    elif m < target_dim:
        # Pad with zeros
        pad_width = target_dim - m
        return np.concatenate([X, np.zeros((num_samples, pad_width))], axis=1)
    else:
        return X
    

# -------------------------------
# Flask Routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/simulate')
def simulate():
    return render_template('simulate.html')

@app.route('/simulate_own')
def simulate_own():
    return render_template('simulate_own.html')

@app.route('/choose')
def choose():
    return render_template('choose.html')

# Existing simulation route for classification
@app.route('/start_simulation', methods=['POST'])
def start_simulation():
    # Extract basic parameters from the simulation form
    fl_type       = request.form.get('fl_type')
    dataset       = request.form.get('dataset')
    num_clients   = int(request.form.get('num_clients'))
    data_type     = 'iid'
    local_epochs  = int(request.form.get('local_epochs'))
    global_epochs = int(request.form.get('global_epochs'))
    num_qubits    = int(request.form.get('num_qubits', 10))
    num_layers    = int(request.form.get('num_layers', 1))
    
    # Extract additional quantum options
    quantum_encoding = request.form.get('quantum_encoding', 'amplitude')
    ansatz           = request.form.get('ansatz', 'ry_cx')
    measurement      = request.form.get('measurement', 'probability')
    differential     = request.form.get('differential', 'parameter_shift')
    loss_function    = request.form.get('loss_function', 'cross_entropy')
    optimizer_choice = request.form.get('optimizer', 'adam')
    learning_rate    = float(request.form.get('learning_rate', 0.1))
    
    # Check for quantum requirements on dataset:
    if fl_type == "quantum_federated" and dataset in ["cifar10", "cifar100"] and num_qubits < 2:
        error_message = ("For CIFAR-10 or CIFAR-100, Quantum Federated Learning requires at least 2 qubits. "
                         "Please choose 2 or more qubits.")
        return render_template("error.html", error_message=error_message)
    
    return render_template('simulation_results.html', 
                           fl_type=fl_type,
                           dataset=dataset,
                           num_clients=num_clients,
                           data_type=data_type,
                           local_epochs=local_epochs,
                           global_epochs=global_epochs,
                           num_qubits=num_qubits,
                           num_layers=num_layers,
                           quantum_encoding=quantum_encoding,
                           ansatz=ansatz,
                           measurement=measurement,
                           differential=differential,
                           loss_function=loss_function,
                           optimizer_choice=optimizer_choice,
                           learning_rate=learning_rate)

@app.route('/stream_simulation')
def stream_simulation():
    # (Classification simulation streaming route remains the same)
    fl_type       = request.args.get('fl_type')
    dataset_name  = request.args.get('dataset')
    num_clients   = int(request.args.get('num_clients'))
    local_epochs  = int(request.args.get('local_epochs'))
    global_epochs = int(request.args.get('global_epochs'))
    num_qubits    = int(request.args.get('num_qubits', 10))
    num_layers    = int(request.args.get('num_layers', 1))
    learning_rate = float(request.args.get('learning_rate', 0.1))
    
    quantum_encoding = request.args.get('quantum_encoding', 'amplitude')
    ansatz           = request.args.get('ansatz', 'ry_cx')
    measurement      = request.args.get('measurement', 'probability')
    differential     = request.args.get('differential', 'parameter_shift')
    loss_function    = request.args.get('loss_function', 'cross_entropy')
    optimizer_choice = request.args.get('optimizer', 'adam')

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    }

    def event_stream_classical():
        if dataset_name == 'mnist':
            input_channels = 1
            image_size = 28
            num_classes = 10
            transform = transforms.Compose([transforms.ToTensor()])
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'fashion_mnist':
            input_channels = 1
            image_size = 28
            num_classes = 10
            transform = transforms.Compose([transforms.ToTensor()])
            trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'cifar100':
            input_channels = 3
            image_size = 32
            num_classes = 10
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        elif dataset_name == 'cifar10':
            input_channels = 3
            image_size = 32
            num_classes = 100
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
            ])
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
        else:
            yield f"data: {json.dumps({'error': 'Invalid dataset selected.'})}\n\n"
            return

        client_indices = partition_dataset(trainset, num_clients)
        client_loaders = []
        batch_size = 32
        for indices in client_indices:
            subset = torch.utils.data.Subset(trainset, indices)
            loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
            client_loaders.append(loader)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        global_model = SimpleCNN(input_channels, num_classes, image_size).to(device)
        criterion = nn.CrossEntropyLoss()

        results = []
        for epoch in range(global_epochs):
            local_state_dicts = []
            for loader in client_loaders:
                local_model = SimpleCNN(input_channels, num_classes, image_size).to(device)
                local_model.load_state_dict(global_model.state_dict())
                optimizer = optim.SGD(local_model.parameters(), lr=learning_rate)
                local_model.train()
                for _ in range(local_epochs):
                    for data, target in loader:
                        data, target = data.to(device), target.to(device)
                        optimizer.zero_grad()
                        outputs = local_model(data)
                        loss = criterion(outputs, target)
                        loss.backward()
                        optimizer.step()
                local_state_dicts.append(copy.deepcopy(local_model.state_dict()))
            global_state_dict = average_weights(local_state_dicts)
            global_model.load_state_dict(global_state_dict)
            test_loss, test_accuracy = evaluate(global_model, test_loader, criterion)
            result = {'global_epoch': epoch + 1, 'loss': test_loss, 'accuracy': test_accuracy}
            results.append(result)
            time.sleep(1)
            yield f"data: {json.dumps(result)}\n\n"
            print(f"Classical Epoch {epoch+1}/{global_epochs} - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")
        yield f"data: {json.dumps({'complete': True})}\n\n"
        global last_simulation_results
        last_simulation_results = results

    if fl_type == "federated":
        stream = event_stream_classical()
    elif fl_type == "quantum_federated":
        stream = event_stream_quantum(
    dataset_name, num_clients, local_epochs, global_epochs,
    num_qubits, num_layers, learning_rate
)
    else:
        stream = iter([f"data: {json.dumps({'error': 'Invalid FL type selected.'})}\n\n"])
    
    return Response(stream, headers=headers)

def event_stream_quantum(
    dataset_name, num_clients, local_epochs, global_epochs,
    num_qubits, num_layers, learning_rate
):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    
    if dataset_name in ['mnist', 'fashion_mnist']:
        if dataset_name == 'mnist':
            from torchvision.datasets import MNIST
            trainset = MNIST(root="MNIST", train=True, download=True, transform=transform)
            testset = MNIST(root="MNIST", train=False, download=True, transform=transform)
        else:
            from torchvision.datasets import FashionMNIST
            trainset = FashionMNIST(root="FashionMNIST", train=True, download=True, transform=transform)
            testset = FashionMNIST(root="FashionMNIST", train=False, download=True, transform=transform)
        global_model = QNN_MNIST(n=num_qubits, L=num_layers).to(device)

    elif dataset_name in ['cifar10', 'cifar100']:
        from torchvision.datasets import CIFAR10, CIFAR100
        if dataset_name == 'cifar10':
            trainset = CIFAR10(root="CIFAR10", train=True, download=True, transform=transform)
            testset = CIFAR10(root="CIFAR10", train=False, download=True, transform=transform)
        else:
            trainset = CIFAR100(root="CIFAR100", train=True, download=True, transform=transform)
            testset = CIFAR100(root="CIFAR100", train=False, download=True, transform=transform)
        global_model = QNN_CIFAR(n=num_qubits, L=num_layers).to(device)

    else:
        yield f"data: {json.dumps({'error': 'Unsupported dataset for quantum FL.'})}\n\n"
        return

    # Limit data for fast training
    frac = 0.1
    from torch.utils.data import random_split
    trainset, _ = random_split(trainset, [int(frac * len(trainset)), len(trainset) - int(frac * len(trainset))])
    testset, _ = random_split(testset, [int(frac * len(testset)), len(testset) - int(frac * len(testset))])

    client_indices = partition_dataset(trainset, num_clients)
    clients_data = [torch.utils.data.Subset(trainset, indices) for indices in client_indices]

    batch_size = 64
    weight_decay_ = 1e-10
    loss_fn = nn.CrossEntropyLoss()
    results = []

    for epoch in range(global_epochs):
        local_state_dicts = []
        for client_dataset in clients_data:
            local_model = type(global_model)(n=num_qubits, L=num_layers).to(device)
            local_model.load_state_dict(global_model.state_dict())
            optimizer = torch.optim.Adam(local_model.parameters(), lr=learning_rate, weight_decay=weight_decay_)
            local_model.train()
            dataloader = torch.utils.data.DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            for _ in range(local_epochs):
                for X, y in dataloader:
                    X, y = X.to(device), y.to(device)
                    out = local_model(X)
                    loss = loss_fn(out, y)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            local_state_dicts.append(copy.deepcopy(local_model.state_dict()))

        # Aggregate and update global model
        global_state_dict = average_weights(local_state_dicts)
        global_model.load_state_dict(global_state_dict)

        # Evaluation
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        global_model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = global_model(X)
                loss = loss_fn(out, y)
                test_loss += loss.item() * X.size(0)
                _, predicted = torch.max(out, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_loss = test_loss / total
        accuracy = correct / total
        result = {'global_epoch': epoch + 1, 'loss': avg_loss, 'accuracy': accuracy}
        results.append(result)
        time.sleep(1)
        yield f"data: {json.dumps(result)}\n\n"
        print(f"Quantum Epoch {epoch+1}/{global_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    yield f"data: {json.dumps({'complete': True})}\n\n"
    global last_simulation_results
    last_simulation_results = results

# -------------------------------
# New Route: Get Headers from Uploaded CSV (for regression simulation)
# -------------------------------
@app.route('/get_headers', methods=['POST'])
def get_headers():
    if 'upload_dataset' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['upload_dataset']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        df = pd.read_csv(file)
        headers = df.columns.tolist()
        # Save the CSV temporarily for later use in regression simulation.
        df.to_csv("temp_uploaded_data.csv", index=False)
        return jsonify({'headers': headers})
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return jsonify({'error': 'Failed to read CSV file'}), 500

# -------------------------------
# New Route: Handle Feature Selection from Modal and start regression simulation
# -------------------------------
@app.route('/start_simulation_custom_final', methods=['POST'])
def start_simulation_custom_final():
    # Retrieve the selected feature options from the modal form.
    selected_features = {}
    for key in request.form:
        if key.startswith("include_"):
            col_name = key[len("include_"):]
            selected_features[col_name] = request.form.get(key)
    
    output_feature = request.form.get("output_feature")
    
    # Store the feature configuration globally for regression simulation.
    global selected_feature_config
    selected_feature_config = {
        "selected_features": selected_features,  # dict mapping column -> "yes"/"no"
        "output_feature": output_feature
    }
    
    # Redirect to the regression simulation results page.
    return redirect(url_for('simulation_regression'))

# -------------------------------
# New Route: Render Regression Simulation Results Page
# -------------------------------
@app.route('/simulation_regression')
def simulation_regression():
    global selected_feature_config
    # Pass selected_feature_config to the template context.
    return render_template('simulation_results_regression.html', selected_feature_config=selected_feature_config)

# -------------------------------
# New Route: Stream Regression Simulation (Quantum FL for regression)
# -------------------------------
@app.route('/stream_simulation_regression')
def stream_simulation_regression():
    # Load regression configuration from the global variable and temporary CSV
    global selected_feature_config
    if not os.path.exists("temp_uploaded_data.csv") or not selected_feature_config:
        return "No data or configuration available.", 400

    # Load the CSV
    df = pd.read_csv("temp_uploaded_data.csv")
    # Determine which columns are to be used as input features (marked "yes")
    input_columns = [col for col, inc in selected_feature_config["selected_features"].items() if inc == "yes"]
    output_column = selected_feature_config["output_feature"]
    
    # Check that the output column is in the dataframe
    if output_column not in df.columns:
        return "Output column not found in data.", 400

    # Extract input and output data (assumes numeric data)
    X = df[input_columns].values.astype(np.float32)
    y = df[output_column].values.astype(np.float32)
    y = y.reshape(-1, 1)  # Ensure y is a column vector

    # Determine quantum parameters from query parameters (or set defaults)
    num_clients = int(request.args.get('num_clients', 3))
    local_epochs = int(request.args.get('local_epochs', 1))
    global_epochs = int(request.args.get('global_epochs', 100))  # Updated default to 100
    num_qubits = int(request.args.get('num_qubits', 10))
    num_layers = int(request.args.get('num_layers', 1))
    learning_rate = float(request.args.get('learning_rate', 0.01))
    
    # Pad feature dimension so that each sample has dimension 2^num_qubits
    X_padded = pad_features(X, num_qubits)
    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_padded)
    y_tensor = torch.tensor(y)

    # Create a simple dataset (assumes data is small enough)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    # Partition data among clients (IID partitioning here)
    client_indices = partition_dataset(dataset, num_clients)
    client_datasets = [torch.utils.data.Subset(dataset, indices) for indices in client_indices]

    batch_size = 16
    # For simplicity, use the last client as test data.
    test_dataset = client_datasets[-1]
    train_datasets = client_datasets[:-1]

    # Define the global QNN regressor model
    global_model = QNN_Regressor(n=num_qubits, L=num_layers).to(device)
    criterion = nn.MSELoss()

    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no"
    }

    def event_stream_regression():
        results = []
        for epoch in range(global_epochs):
            local_state_dicts = []
            for ds in train_datasets:
                local_model = QNN_Regressor(n=num_qubits, L=num_layers).to(device)
                local_model.load_state_dict(global_model.state_dict())
                optimizer = optim.Adam(local_model.parameters(), lr=learning_rate)
                local_model.train()
                dataloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
                for _ in range(local_epochs):
                    for X_batch, y_batch in dataloader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        optimizer.zero_grad()
                        outputs = local_model(X_batch)
                        loss = criterion(outputs, y_batch)
                        loss.backward()
                        optimizer.step()
                local_state_dicts.append(copy.deepcopy(local_model.state_dict()))
            # Aggregate weights
            global_state_dict = average_weights(local_state_dicts)
            global_model.load_state_dict(global_state_dict)

            # Evaluate on test set
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            global_model.eval()
            total_loss = 0.0
            total_samples = 0
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = global_model(X_batch)
                    loss = criterion(outputs, y_batch)
                    total_loss += loss.item() * X_batch.size(0)
                    total_samples += X_batch.size(0)
            avg_loss = total_loss / total_samples
            result = {'global_epoch': epoch + 1, 'mse': avg_loss}
            results.append(result)
            time.sleep(1)
            yield f"data: {json.dumps(result)}\n\n"
            print(f"Regression Epoch {epoch+1}/{global_epochs} - MSE: {avg_loss:.4f}")
        yield f"data: {json.dumps({'complete': True})}\n\n"
    
    return Response(event_stream_regression(), headers=headers)

@app.route('/download_csv')
def download_csv():
    global last_simulation_results
    if not last_simulation_results:
        return "No simulation results available.", 400
    import io, csv
    si = io.StringIO()
    writer = csv.writer(si)
    writer.writerow(['Global Epoch', 'Loss/Accuracy'])
    for res in last_simulation_results:
        writer.writerow([res['global_epoch'], res.get('loss', res.get('mse'))])
    output = si.getvalue()
    return Response(
        output,
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=simulation_results.csv"}
    )

if __name__ == '__main__':
    app.run(debug=True)
