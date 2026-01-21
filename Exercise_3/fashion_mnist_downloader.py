import gzip
import urllib.request
import numpy as np
import os


def download_fashion_mnist():
    """
    Downloads the Fashion-MNIST dataset (training and test data/labels)
    Only downloads files if they don't already exist
    """

    # Base URL for Fashion-MNIST dataset
    base_url = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

    # File names
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }

    # Create directory for dataset
    os.makedirs('fashion_mnist_data', exist_ok=True)

    # Check if all files already exist
    all_files_exist = all(
        os.path.exists(os.path.join('fashion_mnist_data', filename))
        for filename in files.values()
    )

    if all_files_exist:
        print("Skipping download...")
    else:
        print("Downloading Fashion-MNIST dataset...")

        # Download missing files
        for key, filename in files.items():
            filepath = os.path.join('fashion_mnist_data', filename)

            if os.path.exists(filepath):
                print(f"✓ {filename} already exists, skipping...")
            else:
                url = base_url + filename
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(url, filepath)
                print(f"✓ {filename} downloaded")

    # Load training images
    with gzip.open('fashion_mnist_data/train-images-idx3-ubyte.gz', 'rb') as f:
        train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    # Load training labels
    with gzip.open('fashion_mnist_data/train-labels-idx1-ubyte.gz', 'rb') as f:
        train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    # Load test images
    with gzip.open('fashion_mnist_data/t10k-images-idx3-ubyte.gz', 'rb') as f:
        test_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)

    # Load test labels
    with gzip.open('fashion_mnist_data/t10k-labels-idx1-ubyte.gz', 'rb') as f:
        test_labels = np.frombuffer(f.read(), np.uint8, offset=8)

    print(f"Dataset loaded successfully!")
    if not all_files_exist:
        print(f"Training images: {train_images.shape}")
        print(f"Training labels: {train_labels.shape}")
        print(f"Test images: {test_images.shape}")
        print(f"Test labels: {test_labels.shape}")

    print("\n")

    # Class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    return train_images, train_labels, test_images, test_labels, class_names


if __name__ == "__main__":
    # This allows the script to be executed directly
    train_images, train_labels, test_images, test_labels, class_names = download_fashion_mnist()

    print(f"\nClass names: {class_names}")
    print(f"\nData ready to use!")
    print("You can import this function in other scripts:")
    print("  from fashion_mnist_downloader import download_fashion_mnist")
    print("  train_images, train_labels, test_images, test_labels, class_names = download_fashion_mnist()")
