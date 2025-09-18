import torch
from torch import nn
from torch import optim
from torch.utils import data
from dataset_competition import DatasetCompetition
from models import Model
from tensorboardX import SummaryWriter
import os
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--epoch', default=500, type=int, help='train epochs')
    parser.add_argument('--data_path', default='./dataset', type=str, help='path to competition dataset')
    parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate')
    parser.add_argument('--load', default=False, action='store_true', help='load checkpoint')
    parser.add_argument('--save_path', default='./checkpoints_competition', type=str, help='path to save checkpoint')
    parser.add_argument('--gpu', default=0, type=int, help='gpu id')
    parser.add_argument('--log_path', default='./logs_competition', type=str, help='path to log')
    parser.add_argument('--val_freq', default=5, type=int, help='validation frequency (epochs)')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    
    print("=== SFANet Competition Training ===")
    print(f"Dataset path: {args.data_path}")
    print(f"Batch size: {args.bs}")
    print(f"Learning rate: {args.lr}")
    print(f"Total epochs: {args.epoch}")
    print(f"Save path: {args.save_path}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = DatasetCompetition(args.data_path, is_train=True, is_test=False)
    val_dataset = DatasetCompetition(args.data_path, is_train=False, is_test=False)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = data.DataLoader(train_dataset, batch_size=args.bs, shuffle=True, 
                                   num_workers=4, pin_memory=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                 num_workers=2, pin_memory=True)
    
    # Setup device
    device = torch.device('cuda:' + str(args.gpu))
    print(f"Using device: {device}")
    
    # Create model
    model = Model().to(device)
    
    # Setup logging
    writer = SummaryWriter(args.log_path)
    
    # Loss functions (same as original)
    mseloss = nn.MSELoss(reduction='sum').to(device)
    bceloss = nn.BCELoss(reduction='sum').to(device)
    
    # Optimizer 
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Load checkpoint if requested
    if args.load:
        checkpoint_path = os.path.join(args.save_path, 'checkpoint_latest.pth')
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            best_mae = torch.load(os.path.join(args.save_path, 'checkpoint_best.pth'))['mae']
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resumed from epoch {start_epoch}, best MAE: {best_mae:.4f}")
        else:
            print(f"No checkpoint found at {checkpoint_path}, starting from scratch")
            best_mae = 999999
            start_epoch = 0
    else:
        best_mae = 999999
        start_epoch = 0
    
    print(f"\nStarting training from epoch {start_epoch}")
    print("=" * 60)
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + args.epoch):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        loss_avg = 0.0
        loss_att_avg = 0.0
        
        for i, (images, density, att) in enumerate(train_loader):
            images = images.to(device)
            density = density.to(device)
            att = att.to(device)
            
            # Forward pass
            outputs, attention = model(images)
            
            # Calculate losses
            loss_density = mseloss(outputs, density) / args.bs
            loss_attention = bceloss(attention, att) / args.bs * 0.1  # Same weight as original
            loss_total = loss_density + loss_attention
            
            # Backward pass
            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            
            # Accumulate losses
            loss_avg += loss_density.item()
            loss_att_avg += loss_attention.item()
            
            # Print progress
            if (i + 1) % 50 == 0:
                avg_density_loss = loss_avg / (i + 1)
                avg_att_loss = loss_att_avg / (i + 1)
                pred_count = outputs.sum().item() / args.bs
                true_count = density.sum().item() / args.bs
                
                print(f"Epoch {epoch:3d}, Step {i+1:4d}/{len(train_loader)}: "
                      f"Loss_D={avg_density_loss:.4f}, Loss_A={avg_att_loss:.4f}, "
                      f"Pred={pred_count:.1f}, True={true_count:.1f}")
        
        # Calculate epoch averages
        epoch_loss_density = loss_avg / len(train_loader)
        epoch_loss_att = loss_att_avg / len(train_loader)
        epoch_time = time.time() - epoch_start_time
        
        # Log training metrics
        writer.add_scalar('loss/train_loss_density', epoch_loss_density, epoch)
        writer.add_scalar('loss/train_loss_attention', epoch_loss_att, epoch)
        writer.add_scalar('loss/train_loss_total', epoch_loss_density + epoch_loss_att, epoch)
        
        print(f"Epoch {epoch} completed in {epoch_time:.1f}s - "
              f"Avg Loss: {epoch_loss_density:.4f} + {epoch_loss_att:.4f} = {epoch_loss_density + epoch_loss_att:.4f}")
        
        # Validation phase
        if (epoch + 1) % args.val_freq == 0:
            print("Running validation...")
            model.eval()
            
            with torch.no_grad():
                mae = 0.0
                mse = 0.0
                val_count = 0
                
                for images, gt in val_loader:
                    images = images.to(device)
                    
                    # Forward pass
                    predict, _ = model(images)
                    
                    # Calculate metrics
                    pred_count = predict.sum().item()
                    true_count = gt.item()
                    
                    mae += abs(pred_count - true_count)
                    mse += (pred_count - true_count) ** 2
                    val_count += 1
                    
                    if val_count <= 5:  # Print first few predictions
                        print(f"  Val sample {val_count}: Pred={pred_count:.1f}, True={true_count}")
                
                # Calculate final metrics
                mae = mae / len(val_loader)
                mse = (mse / len(val_loader)) ** 0.5
                
                print(f"Validation Results - MAE: {mae:.4f}, MSE: {mse:.4f}")
                
                # Log validation metrics
                writer.add_scalar('eval/MAE', mae, epoch)
                writer.add_scalar('eval/MSE', mse, epoch)
        
        # Save checkpoints
        state = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'mae': mae if (epoch + 1) % args.val_freq == 0 else best_mae,
            'mse': mse if (epoch + 1) % args.val_freq == 0 else 0
        }
        
        # Save latest checkpoint
        torch.save(state, os.path.join(args.save_path, 'checkpoint_latest.pth'))
        
        # Save best checkpoint
        if (epoch + 1) % args.val_freq == 0 and mae < best_mae:
            best_mae = mae
            torch.save(state, os.path.join(args.save_path, 'checkpoint_best.pth'))
            print(f"New best model saved! MAE: {best_mae:.4f}")
        
        print("-" * 60)
    
    print("Training completed!")
    print(f"Best MAE achieved: {best_mae:.4f}")
    writer.close()

if __name__ == '__main__':
    main()