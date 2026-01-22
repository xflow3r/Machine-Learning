import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import load_dataset, create_dataloaders
from utils import set_seed, save_results_to_csv
from metrics import compute_metrics, plot_confusion_matrix
from models.traditional import run_traditional
from models.deep import run_deep


def main():
    parser = argparse.ArgumentParser(description='Fashion-MNIST and CIFAR-10 Classification')
    parser.add_argument('--model', type=str, default='all',
                        choices=['hist_svm', 'hist_logreg', 'hog_svm', 'hog_logreg',
                                 'cnn_small', 'cnn_medium', 'all'],
                        help='Model to run')
    parser.add_argument('--dataset', type=str, default='both',
                        choices=['fashion_mnist', 'cifar10', 'both'],
                        help='Dataset to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for deep learning')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for deep learning models')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for deep learning models')
    parser.add_argument('--augment', action='store_true',
                        help='Use data augmentation for deep learning models')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Determine which datasets to run
    if args.dataset == 'both':
        datasets_to_run = ['fashion_mnist', 'cifar10']
    else:
        datasets_to_run = [args.dataset]

    # Determine which models to run
    if args.model == 'all':
        models_to_run = ['hist_svm', 'hist_logreg', 'hog_svm', 'hog_logreg',
                         'cnn_small', 'cnn_medium']
    else:
        models_to_run = [args.model]

    # Run experiments for each dataset
    for dataset_name in datasets_to_run:
        print("\n" + "=" * 80)
        print(f"DATASET: {dataset_name.upper()}")
        print("=" * 80)

        # Load data
        print(f"Loading {dataset_name} dataset...")
        X_train, y_train, X_test, y_test, class_names = load_dataset(dataset_name)

        # Determine input channels
        is_grayscale = (len(X_train.shape) == 3)
        input_channels = 1 if is_grayscale else 3

        print(f"Training samples: {X_train.shape}")
        print(f"Test samples: {X_test.shape}")
        print(f"Input channels: {input_channels} ({'grayscale' if is_grayscale else 'RGB'})")
        print(f"Number of classes: {len(class_names)}")

        # Run each model
        for model_name in models_to_run:
            print("\n" + "=" * 80)
            print(f"Running {model_name.upper()} on {dataset_name.upper()}")
            if model_name.startswith('cnn') and args.augment:
                print("WITH DATA AUGMENTATION")
            print("=" * 80)

            try:
                if model_name in ['hist_svm', 'hist_logreg', 'hog_svm', 'hog_logreg']:
                    # Traditional methods
                    results = run_traditional(model_name, X_train, y_train, X_test, y_test)

                elif model_name in ['cnn_small', 'cnn_medium']:
                    # Deep learning methods
                    train_loader, test_loader = create_dataloaders(
                        X_train, y_train, X_test, y_test,
                        batch_size=args.batch_size,
                        augment=args.augment,
                        dataset_name=dataset_name
                    )
                    results = run_deep(model_name, train_loader, test_loader,
                                       device=args.device, epochs=args.epochs,
                                       lr=args.lr, input_channels=input_channels)

                # Extract results
                y_true = results['y_true']
                y_pred = results['y_pred']

                # Compute metrics
                metrics = compute_metrics(y_true, y_pred)
                results.update(metrics)

                # Print results
                print(f"\n{'=' * 80}")
                print(f"RESULTS FOR {model_name.upper()} on {dataset_name.upper()}")
                print(f"{'=' * 80}")
                print(f"  Accuracy:   {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
                print(f"  F1-macro:   {results['f1_macro']:.4f}")
                print(f"  Train time: {results['train_time']:.2f}s")
                print(f"  Test time:  {results['test_time']:.2f}s")
                if 'feature_time' in results:
                    print(f"  Feature extraction time: {results['feature_time']:.2f}s")
                    total_time = results['feature_time'] + results['train_time'] + results['test_time']
                else:
                    total_time = results['train_time'] + results['test_time']
                print(f"  Total time: {total_time:.2f}s")
                print(f"{'=' * 80}")

                # Plot confusion matrix
                augment_str = '_augmented' if (model_name.startswith('cnn') and args.augment) else ''
                cm_path = f'results/figures/cm_{dataset_name}_{model_name}{augment_str}.png'
                plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

                # Prepare results for CSV
                csv_results = {
                    'dataset': dataset_name,
                    'model': model_name,
                    'accuracy': results['accuracy'],
                    'f1_macro': results['f1_macro'],
                    'feature_time': results.get('feature_time', 0.0),
                    'train_time': results['train_time'],
                    'test_time': results['test_time'],
                    'total_time': total_time,
                    'seed': args.seed,
                    'augmented': args.augment if model_name.startswith('cnn') else False,
                    'notes': f"epochs={args.epochs},lr={args.lr}" if model_name.startswith('cnn') else ''
                }

                # Save to CSV
                save_results_to_csv(csv_results)

            except Exception as e:
                print(f"❌ Error running {model_name} on {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"Results saved to: results/tables/results.csv")
    print(f"Confusion matrices saved to: results/figures/")
    print("=" * 80)


if __name__ == '__main__':
    main()