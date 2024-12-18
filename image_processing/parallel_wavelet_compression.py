import numpy as np
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    return np.array(image)

def save_image(image, output_path):
    Image.fromarray(image.astype(np.uint8)).save(output_path)

def haar_transform_1d(data):
    n = len(data)
    output = np.zeros(n)
    for i in range(0, n, 2):
        output[i // 2] = (data[i] + data[i + 1]) / np.sqrt(2)
        output[n // 2 + i // 2] = (data[i] - data[i + 1]) / np.sqrt(2)
    return output

def inverse_haar_transform_1d(data):
    n = len(data)
    output = np.zeros(n)
    for i in range(0, n // 2):
        output[2 * i] = (data[i] + data[n // 2 + i]) / np.sqrt(2)
        output[2 * i + 1] = (data[i] - data[n // 2 + i]) / np.sqrt(2)
    return output

def haar_transform_2d(image):
    rows, cols = image.shape
    coeffs = np.zeros((rows, cols))

    # Apply 1D Haar transform to rows
    for i in range(rows):
        coeffs[i, :] = haar_transform_1d(image[i, :])

    # Apply 1D Haar transform to columns
    for j in range(cols):
        coeffs[:, j] = haar_transform_1d(coeffs[:, j])

    return coeffs

def inverse_haar_transform_2d(coeffs):
    rows, cols = coeffs.shape
    image = np.zeros((rows, cols))

    # Apply inverse 1D Haar transform to columns
    for j in range(cols):
        image[:, j] = inverse_haar_transform_1d(coeffs[:, j])

    # Apply inverse 1D Haar transform to rows
    for i in range(rows):
        image[i, :] = inverse_haar_transform_1d(image[i, :])

    return image

def quantize_coefficients(coeffs, threshold):
    coeffs = np.where(np.abs(coeffs) < threshold, 0, coeffs)
    return coeffs

def main(image_path, output_path, threshold):
    image = load_image(image_path)
    coeffs = haar_transform_2d(image)
    quantized_coeffs = quantize_coefficients(coeffs, threshold)
    reconstructed_image = inverse_haar_transform_2d(quantized_coeffs)
    save_image(reconstructed_image, output_path)
import time
if __name__ == "__main__":
    time_values=[]
    t0=time.time()
    main('compression_images/aaaaaa.jpg', 'compression_image/WTcompressed_aaaaaa.jpg', threshold=10)
    t1=time.time()
    time_values.append(t1-t0)
    t0=time.time()
    main('compression_images/image1.jpg', 'compression_image/WTcompressed_image11.jpg', threshold=10)
    t1=time.time()
    time_values.append(t1-t0)
    t0=time.time()
    main('compression_images/image2.jpg', 'compression_image/WTcompressed_image21.jpg', threshold=10)
    t1=time.time()
    time_values.append(t1-t0)
    t0=time.time()
    main('compression_images/image3.jpg', 'compression_image/WTcompressed_image31.jpg', threshold=10)
    t1=time.time()
    time_values.append(t1-t0)
    t0=time.time()
    main('compression_images/image4.jpg', 'compression_image/WTcompressed_image41.jpg', threshold=10)
    t1=time.time()
    time_values.append(t1-t0)
    t0=time.time()
    main('compression_images/image5.jpg', 'compression_image/WTcompressed_image51.jpg', threshold=10)
    t1=time.time()
    time_values.append(t1-t0)
    
