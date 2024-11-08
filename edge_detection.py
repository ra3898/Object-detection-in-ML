# Importing standard Qiskit libraries and configuring account
from qiskit import *
from qiskit.providers.ibmq import IBMQ
from qiskit.compiler import transpile, assemble
from qiskit.visualization import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('bmh')
from PIL import Image

def plot_image(img, title: str):
    plt.title(title)
    plt.xticks(range(img.shape[0]))
    plt.yticks(range(img.shape[1]))
    plt.imshow(img, extent=[0, img.shape[0], img.shape[1], 0], cmap='viridis')
    plt.show()

def amplitude_encode(img_data):
    # Calculate the RMS value
    rms = np.sqrt(np.sum(np.sum(img_data**2, axis=1)))
    # Create normalized image
    image_norm = []
    for arr in img_data:
        for ele in arr:
            image_norm.append(ele / rms)
    # Return the normalized image as a numpy array
    return np.array(image_norm)

style.use('default')

image_size = 128       # Original image-width
image_crop_size = 32   # Width of each part of image for processing

# Load the image from filesystem
image_raw = np.array(Image.open("TrainingImage/User.1.1.jpg"))  # Update the path if necessary
print('Raw Image info:', image_raw.shape)
print('Raw Image datatype:', image_raw.dtype)

image = []
# Use the actual height of the image for the outer loop
for i in range(image_raw.shape[0]):  
    image.append([])
    for j in range(image_raw.shape[1]):
        image[i].append(image_raw[i][j][0] / 255)

image = np.array(image)
print('Image shape (numpy array):', image.shape)

plt.title('Input Image')
plt.xticks(range(0, image.shape[0]+1, 32))
plt.yticks(range(0, image.shape[1]+1, 32))
plt.imshow(image, extent=[0, image.shape[0], image.shape[1], 0], cmap='gray_r')
plt.show()

# Initialize some global variable for number of qubits
data_qb = 10
anc_qb = 1
total_qb = data_qb + anc_qb

# Initialize the amplitude permutation unitary
D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)

new_image = image.copy()  # Create a copy of the image to avoid modifying the original
for i in range (0, image.shape[0], 32):  # Iterate over the height of the image
    print(i)
    for j in range (0, image.shape[1], 32):  # Iterate over the width of the image
        image1 = image[i:i+32, j:j+32]
        image1_norm_h = amplitude_encode(image1)
        image1_norm_v = amplitude_encode(image1.T)
        # Create the circuit for horizontal scan
        qc_h = QuantumCircuit(total_qb)
        qc_h.initialize(image1_norm_h, range(1, total_qb))
        qc_h.h(0)
        qc_h.unitary(D2n_1, range(total_qb))
        qc_h.h(0)
        # display(qc_h.draw('mpl', fold=-1))

        # Create the circuit for vertical scan
        qc_v = QuantumCircuit(total_qb)
        qc_v.initialize(image1_norm_v, range(1, total_qb))
        qc_v.h(0)
        qc_v.unitary(D2n_1, range(total_qb))
        qc_v.h(0)
        # display(qc_v.draw('mpl', fold=-1))

        # Combine both circuits into a single list
        circ_list = [qc_h, qc_v]

        # Simulating the cirucits
        back = Aer.get_backend('statevector_simulator')
        results = execute(circ_list, backend=back).result()
        sv_h = results.get_statevector(qc_h)
        sv_v = results.get_statevector(qc_v)
        threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)

        # Selecting odd states from the raw statevector and
        # reshaping column vector of size 64 to an 8x8 matrix
        edge_scan_h = np.abs(np.array([1 if threshold(sv_h[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(32, 32)
        edge_scan_v = np.abs(np.array([1 if threshold(sv_v[2*i+1].real) else 0 for i in range(2**data_qb)])).reshape(32, 32).T

        edge_scan_sim = edge_scan_h | edge_scan_v

        for a in range (32):
            for b in range (32):
                new_image[i+a][j+b]=edge_scan_sim[a][b]

plt.title('Edge Detected Image')
plt.xticks(range(0, new_image.shape[0]+1, 32))
plt.yticks(range(0, new_image.shape[1]+1, 32))
plt.imshow(new_image, extent=[0, new_image.shape[0], new_image.shape[1], 0], cmap='gray_r')
plt.show()
print(new_image.shape)

data = Image.fromarray(new_image)
plt.imsave('filename.jpeg', data)
