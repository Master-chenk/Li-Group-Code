import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import argparse

from model.U_Net import U_Net
from data.moving_mnist import get_dataloaders
from utils.metrics import calculate_mae, calculate_mse


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for inputs, targets in tqdm(train_loader, desc='train'):
        inputs = inputs.float().to(device) / 255.0
        targets = targets.float().to(device) / 255.0

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, test_loader, criterion, device):

    model.eval()
    total_loss = 0
    total_mae = 0
    total_mse = 0

    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='val'):
            inputs = inputs.float().to(device) / 255.0
            targets = targets.float().to(device) / 255.0

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            total_mae += calculate_mae(outputs, targets)
            total_mse += calculate_mse(outputs, targets)

    avg_loss = total_loss / len(test_loader)
    avg_mae = total_mae / len(test_loader)
    avg_mse = total_mse / len(test_loader)

    return avg_loss, avg_mae, avg_mse


def train_model(model, train_loader, test_loader, criterion, optimizer, device, args):

    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_mse': []}

    print(f"Using device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}, Testing samples: {len(test_loader.dataset)}")
    print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}, Number of epochs: {args.epochs}")


    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')


        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        history['train_loss'].append(train_loss)


        val_loss, val_mae, val_mse = validate(model, test_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_mae'].append(val_mae)
        history['val_mse'].append(val_mse)

        print(f'Train Loss: {train_loss:.6f}')
        print(f'Val Loss: {val_loss:.6f}, MAE: {val_mae:.6f}, MSE: {val_mse:.6f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f'Saved best model (Val Loss: {val_loss:.6f})')


    torch.save(history, 'results/history.pth')


    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot(history['val_mae'])
    axes[1].set_title('MAE')
    axes[1].grid()

    axes[2].plot(history['val_mse'])
    axes[2].set_title('MSE')
    axes[2].grid()

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150)


def visualize_predictions(model, test_loader, device, num_samples=2):
    model.eval()

    inputs, targets = next(iter(test_loader))
    inputs = inputs.float().to(device) / 255.0
    targets = targets.float().to(device) / 255.0

    with torch.no_grad():
        outputs = model(inputs)

    mae = calculate_mae(outputs, targets)
    mse = calculate_mse(outputs, targets)

    print(f'MAE: {mae:.6f}, MSE: {mse:.6f}')

    inputs = inputs.cpu()
    targets = targets.cpu()
    outputs = outputs.cpu()

    os.makedirs('results', exist_ok=True)

    num_samples = min(num_samples, inputs.shape[0])

    num_input_frames = inputs.shape[1] 

    num_output_frames = targets.shape[1] 
    total_cols = num_input_frames + num_output_frames

    fig, axes = plt.subplots(num_samples * 2, total_cols,
                            figsize=(total_cols * 0.8, num_samples * 2 * 0.8))


    if num_samples == 1:
        axes = axes.reshape(2, -1)

    for sample_idx in range(num_samples):
        row_truth = sample_idx * 2     
        row_pred = sample_idx * 2 + 1  

        col = 0


        for t in range(num_input_frames):
            axes[row_truth, col].imshow(inputs[sample_idx, t], cmap='gray', vmin=0, vmax=1)
            axes[row_truth, col].axis('off')
            if sample_idx == 0:
                axes[row_truth, col].set_title(f'T={t+1}', fontsize=9)
            col += 1

        for t in range(num_output_frames):
            axes[row_truth, col].imshow(targets[sample_idx, t], cmap='gray', vmin=0, vmax=1)
            axes[row_truth, col].axis('off')
            if sample_idx == 0:
                axes[row_truth, col].set_title(f'T={num_input_frames + t + 1}', fontsize=9)
            col += 1

        col = 0

        for i in range(num_input_frames):
            axes[row_pred, col].axis('off')
            col += 1

        for t in range(num_output_frames):
            axes[row_pred, col].imshow(outputs[sample_idx, t], cmap='gray', vmin=0, vmax=1)
            axes[row_pred, col].axis('off')
            col += 1

        fig.text(0.02, 1 - (sample_idx * 2 + 0.5) / (num_samples * 2),
                'Truth', ha='right', va='center', fontsize=10, weight='bold')
        fig.text(0.02, 1 - (sample_idx * 2 + 1.5) / (num_samples * 2),
                'Pred', ha='right', va='center', fontsize=10, weight='bold')

    plt.tight_layout(rect=[0.03, 0, 1, 1])
    plt.savefig('results/comparison.png', dpi=150, bbox_inches='tight')
    plt.close()



def main():
    parser = argparse.ArgumentParser(description='Moving MNIST')

    parser.add_argument('--visualize-only', action='store_true',default=True, help='only visualize predictions using a pre-trained model')

    parser.add_argument('--batch-size', type=int, default=32, help='batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers (default: 4)')

    parser.add_argument('--no-visualize', action='store_true', default=False, help='do not visualize after training')
    parser.add_argument('--num-samples', type=int, default=2, help='number of samples to visualize (default: 2)')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # ========== create model ==========
    # 后续修改只需在此处更改模型定义即可, 默认的任务是[B,C_in,H,W] -> [B,C_out,H,W] 

    model = U_Net(input_channel=10, num_classes=10).to(device)
    
    # ========================================================

    if args.visualize_only:
        # only visualize mode - load trained model
        if not os.path.exists('checkpoints/best_model.pth'):
            print("error: 'checkpoints/best_model.pth' not found")
            return

        model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))


        _, test_loader = get_dataloaders(batch_size=max(4, args.num_samples), num_workers=0)

        visualize_predictions(model, test_loader, device, num_samples=args.num_samples)

    else:
        # training mode
        train_loader, test_loader = get_dataloaders(batch_size=args.batch_size, num_workers=args.workers)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        train_model(model, train_loader, test_loader, criterion, optimizer, device, args)

        if not args.no_visualize:
            _, test_loader = get_dataloaders(batch_size=max(4, args.num_samples), num_workers=args.workers)
            visualize_predictions(model, test_loader, device, num_samples=args.num_samples)


if __name__ == '__main__':
    main()
