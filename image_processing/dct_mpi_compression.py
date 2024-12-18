import numpy as np
from mpi4py import MPI

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

BLOCK_SIZE = 8

def dct(block):
    # Implement DCT here
    return np.fft.dct(np.fft.dct(block.T, norm='ortho').T, norm='ortho')

def idct(block):
    # Implement IDCT here
    return np.fft.idct(np.fft.idct(block.T, norm='ortho').T, norm='ortho')

def quantize(block):
    return np.round(block / QUANTIZATION_MAT)

def dequantize(block):
    return block * QUANTIZATION_MAT

def compress_image(image_path, output_path):
    # Load image
    img = np.load(image_path)
    h, w = img.shape

    nbh = (h + BLOCK_SIZE - 1) // BLOCK_SIZE
    nbw = (w + BLOCK_SIZE - 1) // BLOCK_SIZE

    H = nbh * BLOCK_SIZE
    W = nbw * BLOCK_SIZE

    padded_img = np.pad(img, ((0, H - h), (0, W - w)), mode='constant')

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    blocks_per_process = (nbh * nbw) // size
    start_block = rank * blocks_per_process
    end_block = min(start_block + blocks_per_process, nbh * nbw)

    for block_index in range(start_block, end_block):
        i = block_index // nbw
        j = block_index % nbw

        row_ind_1 = i * BLOCK_SIZE
        row_ind_2 = row_ind_1 + BLOCK_SIZE
        col_ind_1 = j * BLOCK_SIZE
        col_ind_2 = col_ind_1 + BLOCK_SIZE

        block = padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2]
        block = dct(block)
        block = quantize(block)

        padded_img[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = block

    if rank == 0:
        compressed_data = np.zeros((H, W), dtype=int)
    else:
        compressed_data = None

    comm.Gather(padded_img, compressed_data, root=0)

    if rank == 0:
        np.save(output_path, compressed_data)

def decompress_image(compressed_path, output_path):
    # Load compressed data
    compressed_data = np.load(compressed_path)
    H, W = compressed_data.shape

    nbh = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
    nbw = (W + BLOCK_SIZE - 1) // BLOCK_SIZE

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    blocks_per_process = (nbh * nbw) // size
    start_block = rank * blocks_per_process
    end_block = min(start_block + blocks_per_process, nbh * nbw)

    for block_index in range(start_block, end_block):
        i = block_index // nbw
        j = block_index % nbw

        row_ind_1 = i * BLOCK_SIZE
        row_ind_2 = row_ind_1 + BLOCK_SIZE
        col_ind_1 = j * BLOCK_SIZE
        col_ind_2 = col_ind_1 + BLOCK_SIZE

        block = compressed_data[row_ind_1:row_ind_2, col_ind_1:col_ind_2]
        block = dequantize(block)
        block = idct(block)

        compressed_data[row_ind_1:row_ind_2, col_ind_1:col_ind_2] = block

    if rank == 0:
        decompressed_img = np.zeros((H, W), dtype=int)
    else:
        decompressed_img = None

    comm.Gather(compressed_data, decompressed_img, root=0)

    if rank == 0:
        np.save(output_path, decompressed_img)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_path> <output_path> <compress|decompress>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    mode = sys.argv[3]

    if mode == "compress":
        compress_image(input_path, output_path)
    elif mode == "decompress":
        decompress_image(input_path, output_path)
    else:
        print("Invalid mode. Use 'compress' or 'decompress'.")
        sys.exit(1)
