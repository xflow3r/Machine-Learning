import argparse
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import load_fashion_mnist, create_dataloaders
from utils import set_seed, save_results_to_csv
from metrics import compute_metrics, plot_confusion_matrix
from models.traditional import run_traditional
from models.deep import run_deep


def main():
    parser = argparse.ArgumentParser(description='Fashion-MNIST Classification')
    parser.add_argument('--model', type=str, default='all',
                        choices=['hog_svm', 'hog_logreg', 'cnn_small', 'cnn_medium', 'all'],
                        help='Model to run')
    parser.add_argument('--dataset', type=str, default='fashion_mnist',
                        help='Dataset to use (currently only fashion_mnist)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for deep learning models')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for deep learning models')

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Load data
    print("=" * 80)
    print("Loading Fashion-MNIST dataset...")
    print("=" * 80)
    X_train, y_train, X_test, y_test, class_names = load_fashion_mnist()

    # Determine which models to run
    if args.model == 'all':
        models_to_run = ['hog_svm', 'hog_logreg', 'cnn_small', 'cnn_medium']
    else:
        models_to_run = [args.model]

    # Run each model
    for model_name in models_to_run:
        print("\n" + "=" * 80)
        print(f"Running {model_name.upper()}")
        print("=" * 80)

        try:
            if model_name in ['hog_svm', 'hog_logreg']:
                # Traditional methods
                results = run_traditional(model_name, X_train, y_train, X_test, y_test)

            elif model_name in ['cnn_small', 'cnn_medium']:
                # Deep learning methods
                train_loader, test_loader = create_dataloaders(
                    X_train, y_train, X_test, y_test, batch_size=args.batch_size
                )
                results = run_deep(model_name, train_loader, test_loader, device=args.device)

            # Extract results
            y_true = results['y_true']
            y_pred = results['y_pred']

            # Compute metrics (if not already in results)
            if 'accuracy' not in results or 'f1_macro' not in results:
                metrics = compute_metrics(y_true, y_pred)
                results.update(metrics)

            # Print results
            print(f"\nResults for {model_name}:")
            print(f"  Accuracy: {results['accuracy']:.4f}")
            print(f"  F1-macro: {results['f1_macro']:.4f}")
            print(f"  Train time: {results['train_time']:.2f}s")
            print(f"  Test time: {results['test_time']:.2f}s")
            if 'feature_time' in results:
                print(f"  Feature extraction time: {results['feature_time']:.2f}s")

            # Plot confusion matrix
            cm_path = f'results/figures/cm_{args.dataset}_{model_name}.png'
            plot_confusion_matrix(y_true, y_pred, class_names, cm_path)

            # Prepare results for CSV
            csv_results = {
                'dataset': args.dataset,
                'model': model_name,
                'accuracy': results['accuracy'],
                'f1_macro': results['f1_macro'],
                'feature_time': results.get('feature_time', 0.0),
                'train_time': results['train_time'],
                'test_time': results['test_time'],
                'total_time': results.get('feature_time', 0.0) + results['train_time'] + results['test_time'],
                'seed': args.seed,
                'notes': ''
            }

            # Save to CSV
            save_results_to_csv(csv_results)

        except NotImplementedError:
            print(f"❌ {model_name} not yet implemented")
        except Exception as e:
            print(f"❌ Error running {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("All experiments completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()