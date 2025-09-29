import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from collections import defaultdict


class Visualizer:
    def __init__(self, class_names=None):
        """
        Visualization utilities for neural network training and evaluation.

        Args:
            class_names (list, optional): List of class names for labeling
        """
        self.class_names = class_names
        plt.style.use('default')

    def plot_training_history(self, history, save_path=None):
        """
        Plot training history including loss and accuracy curves.

        Args:
            history (dict): Training history with keys like 'train_loss', 'val_loss', etc.
            save_path (str, optional): Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot loss
        if 'train_loss' in history:
            axes[0].plot(history['train_loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)

        axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot accuracy
        if 'train_acc' in history:
            axes[1].plot(history['train_acc'], label='Training Accuracy', linewidth=2)
        if 'val_acc' in history:
            axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)

        axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")

        plt.show()

    def plot_sample_images(self, data_loader, num_samples=8, denormalize_fn=None, save_path=None):
        """
        Plot sample images from a data loader.

        Args:
            data_loader: DataLoader to sample from
            num_samples (int): Number of samples to display
            denormalize_fn (callable, optional): Function to denormalize images
            save_path (str, optional): Path to save the plot
        """
        # Get a batch of data
        images, labels = next(iter(data_loader))

        # Select subset of images
        images = images[:num_samples]
        labels = labels[:num_samples]

        # Create subplot grid
        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3*rows))

        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            row = i // cols
            col = i % cols

            # Get image
            img = images[i]

            # Denormalize if function provided
            if denormalize_fn:
                img = denormalize_fn(img)

            # Convert to numpy and transpose for plotting
            img = img.numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)  # Ensure values are in [0, 1]

            # Plot image
            axes[row, col].imshow(img)
            title = f"Label: {labels[i].item()}"
            if self.class_names:
                title += f" ({self.class_names[labels[i].item()]})"
            axes[row, col].set_title(title)
            axes[row, col].axis('off')

        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sample images plot saved to {save_path}")

        plt.show()

    def plot_confusion_matrix(self, model, data_loader, device, save_path=None):
        """
        Plot confusion matrix for model predictions.

        Args:
            model: Trained model
            data_loader: DataLoader for evaluation
            device: Device to run inference on
            save_path (str, optional): Path to save the plot
        """
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, pred = output.max(1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        # Compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

        # Print classification report
        if self.class_names:
            print("\nClassification Report:")
            print(classification_report(all_labels, all_preds, target_names=self.class_names))

    def plot_model_predictions(self, model, data_loader, device, num_samples=8,
                             denormalize_fn=None, save_path=None):
        """
        Plot sample images with model predictions.

        Args:
            model: Trained model
            data_loader: DataLoader to sample from
            device: Device to run inference on
            num_samples (int): Number of samples to display
            denormalize_fn (callable, optional): Function to denormalize images
            save_path (str, optional): Path to save the plot
        """
        model.eval()

        # Get a batch of data
        images, labels = next(iter(data_loader))
        images = images[:num_samples]
        labels = labels[:num_samples]

        # Get predictions
        with torch.no_grad():
            images_gpu = images.to(device)
            outputs = model(images_gpu)
            _, predicted = outputs.max(1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

        predicted = predicted.cpu()
        probabilities = probabilities.cpu()

        # Create subplot grid
        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))

        if rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            row = i // cols
            col = i % cols

            # Get image
            img = images[i]

            # Denormalize if function provided
            if denormalize_fn:
                img = denormalize_fn(img)

            # Convert to numpy and transpose for plotting
            img = img.numpy().transpose(1, 2, 0)
            img = np.clip(img, 0, 1)

            # Plot image
            axes[row, col].imshow(img)

            # Create title with true and predicted labels
            true_label = labels[i].item()
            pred_label = predicted[i].item()
            confidence = probabilities[i][pred_label].item() * 100

            title = f"True: {true_label}"
            if self.class_names:
                title = f"True: {self.class_names[true_label]}"

            title += f"\nPred: {pred_label}"
            if self.class_names:
                title += f" ({self.class_names[pred_label]})"

            title += f"\nConf: {confidence:.1f}%"

            # Color code based on correctness
            color = 'green' if true_label == pred_label else 'red'
            axes[row, col].set_title(title, color=color, fontweight='bold')
            axes[row, col].axis('off')

        # Hide empty subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")

        plt.show()

    def plot_class_distribution(self, class_distribution, title="Class Distribution", save_path=None):
        """
        Plot class distribution as a bar chart.

        Args:
            class_distribution (dict): Dictionary with class names/indices and counts
            title (str): Plot title
            save_path (str, optional): Path to save the plot
        """
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())

        plt.figure(figsize=(12, 6))
        bars = plt.bar(classes, counts, color='skyblue', edgecolor='navy', alpha=0.7)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom')

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Number of Samples')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Class distribution plot saved to {save_path}")

        plt.tight_layout()
        plt.show()

    def plot_learning_rate_schedule(self, lr_history, save_path=None):
        """
        Plot learning rate schedule over training.

        Args:
            lr_history (list): List of learning rates per epoch
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(lr_history, linewidth=2, color='red')
        plt.title('Learning Rate Schedule', fontsize=16, fontweight='bold')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning rate schedule plot saved to {save_path}")

        plt.show()

    def plot_activation_distributions(self, model, data_loader, device, layer_names=None, save_path=None):
        """
        Plot activation distributions for specified layers.

        Args:
            model: Trained model
            data_loader: DataLoader to sample from
            device: Device to run inference on
            layer_names (list, optional): Names of layers to analyze
            save_path (str, optional): Path to save the plot
        """
        model.eval()
        activations = defaultdict(list)

        # Register hooks to capture activations
        def get_activation(name):
            def hook(model, input, output):
                activations[name].append(output.detach().cpu().numpy().flatten())
            return hook

        # Register hooks
        hooks = []
        if layer_names is None:
            # Use all named modules
            layer_names = [name for name, _ in model.named_modules() if len(list(_.children())) == 0]

        for name, layer in model.named_modules():
            if name in layer_names:
                hooks.append(layer.register_forward_hook(get_activation(name)))

        # Run inference on a batch
        with torch.no_grad():
            data, _ = next(iter(data_loader))
            data = data.to(device)
            _ = model(data)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        # Plot distributions
        num_layers = len(activations)
        if num_layers == 0:
            print("No activations captured")
            return

        cols = 3
        rows = (num_layers + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        for i, (name, acts) in enumerate(activations.items()):
            row = i // cols
            col = i % cols

            # Concatenate all activations for this layer
            all_acts = np.concatenate(acts)

            axes[row, col].hist(all_acts, bins=50, alpha=0.7, density=True)
            axes[row, col].set_title(f'{name}')
            axes[row, col].set_xlabel('Activation Value')
            axes[row, col].set_ylabel('Density')
            axes[row, col].grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(num_layers, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')

        plt.suptitle('Activation Distributions', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Activation distributions plot saved to {save_path}")

        plt.show()


def create_visualizer(class_names=None):
    """
    Convenience function to create a visualizer.

    Args:
        class_names (list, optional): List of class names

    Returns:
        Visualizer: Configured visualizer instance
    """
    return Visualizer(class_names=class_names)