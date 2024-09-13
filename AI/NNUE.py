import torch
import torch.nn as nn
import torch.optim as optim
import stockfish
import lichess
from torch.utils.data import DataLoader, TensorDataset
from os import listdir
import chess.pgn


engine = stockfish.Stockfish("stockfish-windows-x86-64-avx2.exe")
games = [f for f in listdir('./game_database')]
game_pgns = []
for game in games:
    game_pgns.append("game_database/"+game)
def pgn_to_fens(pgn_file):
    with open(pgn_file, 'r') as pgn:
        pgn = chess.pgn.read_game(pgn)
        # Create a chess board
        board = pgn.board()
        fens = []
        for move in pgn.mainline_moves():
            fens.append(board.fen())
            board.push(move)
        return fens


def get_stockfish_eval(fen):
    engine.set_fen_position(fen)
    eval = engine.get_evaluation()
    return eval


def generate_datasets(pgn_files):
    fens = []
    evals = []
    for file in pgn_files:
        for fen in pgn_to_fens(file):
            fens.append(fen)
            evals.append(get_stockfish_eval(fen))
    return fens, evals


class NNUE(nn.Module):
    def __init__(self):
        super(NNUE, self).__init__()
        # Flatten the board (8x8) to a vector of 64 elements
        self.fc1 = nn.Linear(64, 128)  # First hidden layer
        self.fc2 = nn.Linear(128, 128)  # Second hidden layer
        self.fc3 = nn.Linear(128, 1)    # Output layer (evaluation score)

    def forward(self, x):
        x = x.view(-1, 64)              # Flatten the board
        x = torch.relu(self.fc1(x))     # Apply ReLU activation
        x = torch.relu(self.fc2(x))     # Second hidden layer with ReLU
        x = self.fc3(x)                 # Output the score
        return x


model = NNUE()
loss_function = nn.MSELoss()  # Mean Squared Error since we are predicting numerical evaluations
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
positions, evaluations = generate_datasets(game_pgns)
dataset = TensorDataset(torch.stack(positions), torch.stack(evaluations))
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)


def train_nnue(model, data_loader, epochs=10):
    # Set the model to training mode
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        # Iterate over the data loader
        for positions, labels in data_loader:
            # Move the data to the appropriate device (e.g., GPU if available)
            positions = positions.to(device)
            labels = labels.to(device)

            # Zero the gradients from the previous step
            optimizer.zero_grad()

            # Forward pass: get the model's output (evaluation) for the input positions
            outputs = model(positions)

            # Calculate the loss between the predicted evaluation and the true labels
            loss = loss_function(outputs.squeeze(), labels)

            # Backward pass: compute gradients
            loss.backward()

            # Update the model weights
            optimizer.step()

            # Track the loss during training
            running_loss += loss.item()

        # Print the average loss for the epoch
        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")


train_nnue(model=model, data_loader=data_loader, epochs=15)
torch.save(model.state_dict(), "nnue_model.pth")


