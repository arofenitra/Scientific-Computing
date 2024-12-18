import cv2
import numpy as np
import zlib
from mpi4py import MPI
import scipy as sp

# Quantization Matrix
QUANTIZATION_MAT = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

# Block size
block_size = 8

def dct_quantize(image):
    h, w = image.shape
    nbh = np.ceil(h / block_size).astype(int)
    nbw = np.ceil(w / block_size).astype(int)

    H = block_size * nbh
    W = block_size * nbw

    padded_img = np.zeros((H, W))
    padded_img[0:h, 0:w] = image[0:h, 0:w]

    for i in range(nbh):
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size

        for j in range(nbw):
            col_ind_1 = j * block_size
            col_ind_2 = col_ind_1 + block_size

            block = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2]
            DCT = cv2.dct(np.float32(block))
            DCT_normalized = np.divide(DCT, QUANTIZATION_MAT).astype(int)
            padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = DCT_normalized

    return padded_img

def idct_dequantize(image):
    h, w = image.shape
    nbh = np.ceil(h / block_size).astype(int)
    nbw = np.ceil(w / block_size).astype(int)

    padded_img = np.zeros((h, w))

    for i in range(nbh):
        row_ind_1 = i * block_size
        row_ind_2 = row_ind_1 + block_size

        for j in range(nbw):
            col_ind_1 = j * block_size
            col_ind_2 = col_ind_1 + block_size

            temp_stream = image[row_ind_1:row_ind_2, col_ind_1:col_ind_2]
            de_quantized = np.multiply(temp_stream, QUANTIZATION_MAT)
            padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = cv2.idct(de_quantized)

    padded_img[padded_img > 255] = 255
    padded_img[padded_img < 0] = 0

    return padded_img

def get_run_length_encoding(image):
    image = image.astype(int)
    stream = []
    i = 0
    while i < image.size:
        count = 0
        while i + 1 < image.size and image[i] == image[i + 1]:
            i += 1
            count += 1
        stream.append((image[i], count))
        i += 1
    return ' '.join(f'{val} {count}' for val, count in stream)

def compress_data(data):
    return zlib.compress(data.encode())

def decompress_data(data):
    return zlib.decompress(data).decode()

def compress_image(image_path, output_path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        chunk_size = h // size
        chunks = [image[i*chunk_size:(i+1)*chunk_size, :] for i in range(size)]
    else:
        chunks = None

    chunk = comm.scatter(chunks, root=0)
    dct_quantized_chunk = dct_quantize(chunk)
    bitstream_chunk = get_run_length_encoding(dct_quantized_chunk.flatten())

    bitstream_chunks = comm.gather(bitstream_chunk, root=0)

    if rank == 0:
        bitstream = ' '.join(bitstream_chunks)
        bitstream = f'{h} {w} {bitstream};'
        compressed_data = compress_data(bitstream)
        with open(output_path, 'wb') as f:
            f.write(compressed_data)

def decompress_image(input_path, output_path):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        with open(input_path, 'rb') as f:
            compressed_data = f.read()
        bitstream = decompress_data(compressed_data)
        details = bitstream.split()
        h = int(details[0])
        w = int(details[1])
        array = np.zeros(h * w).astype(int)
        k = 0
        i = 2
        while k < array.shape[0]:
            if details[i] == ';':
                break
            if "-" not in details[i]:
                array[k] = int(''.join(filter(str.isdigit, details[i])))
            else:
                array[k] = -1 * int(''.join(filter(str.isdigit, details[i])))
            if i + 3 < len(details):
                j = int(''.join(filter(str.isdigit, details[i + 3])))
            if j == 0:
                k += 1
            else:
                k += j + 1
            i += 2
        array = np.reshape(array, (h, w))
        array = array.astype(np.float32)  # Convert to floating-point array
        chunk_size = h // size
        chunks = [array[i*chunk_size:(i+1)*chunk_size, :] for i in range(size)]
    else:
        chunks = None

    chunk = comm.scatter(chunks, root=0)
    decompressed_chunk = idct_dequantize(chunk)
    decompressed_chunks = comm.gather(decompressed_chunk, root=0)

    if rank == 0:
        decompressed_image = np.vstack(decompressed_chunks)
        cv2.imwrite(output_path, np.uint8(decompressed_image))

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Compress the image
        compress_image('compression_images/image1.jpg', 'compressed_image1.bin')

        # Decompress the image
        decompress_image('compressed_image1.bin', 'compression_images/DCTdecompressed_image1.jpeg')
