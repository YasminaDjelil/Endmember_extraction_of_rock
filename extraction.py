"""
MIT License

Copyright (c) [2024] [Djelil Yasmina Feriel]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# necessary libraries
import re
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from matplotlib import patches
from sklearn.cluster import KMeans
import pysptools.eea as eea

# func

def extract_envi_info(header_file_path):
    patterns = {
        "samples": re.compile(r'samples\s*=\s*(\d+)', re.IGNORECASE),
        "bands": re.compile(r'bands\s*=\s*(\d+)', re.IGNORECASE),
        "lines": re.compile(r'lines\s*=\s*(\d+)', re.IGNORECASE),
        "wavelengths": re.compile(r'\d+\.\d+')
    }

    with open(header_file_path, 'r') as file:
        header_text = file.read()

    samples = int(patterns["samples"].search(header_text).group(1))
    bands = int(patterns["bands"].search(header_text).group(1))
    lines = int(patterns["lines"].search(header_text).group(1))
    wavelengths = [float(w) for w in patterns["wavelengths"].findall(header_text)]

    return samples, bands, lines, wavelengths

def load_raw_hyperspectral_data(raw_path, crop_region=None):
    raw_dataset = gdal.Open(raw_path)
    bands = raw_dataset.RasterCount
    lines = raw_dataset.RasterYSize
    samples = raw_dataset.RasterXSize

    raw_data = np.empty((bands, lines, samples), dtype=np.uint16)
    for i in range(bands):
        band_data = raw_dataset.GetRasterBand(i + 1).ReadAsArray()
        raw_data[i, :, :] = band_data

    raw_data = np.transpose(raw_data, (1, 2, 0))

    if crop_region:
        x, y, width, height = crop_region
        raw_data = raw_data[y:y + height, x:x + width, :]

    return raw_data

def em_extraction(data, band_index, pixel_coordinates, wavelengths, a):
    band_data = data[:, :, band_index]

    plt.figure(figsize=(10, 8))
    plt.imshow(band_data, cmap='gist_gray')
    plt.colorbar(label='Pixel Intensity')
    plt.title(f'Data from Band {band_index + 1} (Wavelength: {wavelengths[band_index]} nm)')

    for coord in pixel_coordinates:
        x, y = coord
        plt.scatter(x, y, marker='x', label='Pixel Coordinates')
        x_start, x_end = max(0, int(x - (a / 2) - 1)), min(data.shape[1], int(x + (a / 2)))
        y_start, y_end = max(0, int(y - (a / 2) - 1)), min(data.shape[0], int(y + (a / 2)))
        rect = patches.Rectangle((x_start - 0.5, y_start - 0.5), x_end - x_start, y_end - y_start,
                                  linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=len(pixel_coordinates) // 2)
    plt.show()

    spectra_array = []
    for coord in pixel_coordinates:
        x, y = coord
        x_start, x_end = max(0, int(x - (a / 2) - 1)), min(data.shape[1], int(x + (a / 2)))
        y_start, y_end = max(0, int(y - (a / 2) - 1)), min(data.shape[0], int(y + (a / 2)))
        spectra_square = data[y_start:y_end, x_start:x_end]
        mean_spectrum = np.mean(spectra_square, axis=(0, 1))
        mean_spectrum = (mean_spectrum - np.min(mean_spectrum)) / (np.max(mean_spectrum) - np.min(mean_spectrum))
        spectra_array.append(mean_spectrum)

    plt.figure(figsize=(8, 6))
    plt.title('Spectra of Specified Pixels')
    for i, coord in enumerate(pixel_coordinates):
        x, y = coord
        plt.plot(wavelengths, spectra_array[i], label=f'Pixel {x}, {y} Spectrum')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend(loc='lower center', bbox_to_anchor=(0.9, -0.2), shadow=True, ncol=len(pixel_coordinates) // 2)
    plt.show()

    return np.array(spectra_array)

def perform_kmeans(data, n_clusters):
    if len(data.shape) == 3:
        samples = data.shape[0] * data.shape[1]
        features = data.shape[2]
        data = data.reshape(samples, features)
    elif len(data.shape) != 2:
        raise ValueError("Input data should be a 2D or 3D array")

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(data)
    em_kmeans = kmeans.cluster_centers_

    em_kmeans_normalized = (em_kmeans - np.min(em_kmeans, axis=0)) / (np.max(em_kmeans, axis=0) - np.min(em_kmeans, axis=0))
    return em_kmeans, em_kmeans_normalized

def plot_spectra(endmember_spectra, reconstructed_spectra, wavelengths):

    plt.figure(figsize=(10, 8))
    plt.title('Reconstructed and Original Endmember Spectra')

    for i, spectrum in enumerate(reconstructed_spectra):
        plt.plot(wavelengths, spectrum, label=f'Reconstructed Endmember {i + 1}')

    for j, spectrum in enumerate(endmember_spectra):
        plt.plot(wavelengths, spectrum, '+', label=f'Original Endmember {j + 1}')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_abundance_map(raw_data, em, colors, wavelengths_dim):

    distances = np.linalg.norm(raw_data.reshape(-1, wavelengths_dim, 1) - em.T, axis=1)
    closest_clusters = np.argmin(distances, axis=1)

    abundance_map = closest_clusters.reshape(raw_data.shape[:2])

    cluster_colors = colors

    plt.figure(figsize=(12, 9))
    cmap = ListedColormap(cluster_colors)
    plt.imshow(abundance_map, cmap=cmap, interpolation='nearest')
    plt.axis('off')
    plt.show()



##############################

# files path
raw_path = "/content/drive/MyDrive/Thesis/Yasmina stuff/Rocks/correctedCapture/Corrected.raw"
hdr_path = "/content/drive/MyDrive/Thesis/Yasmina stuff/Rocks/correctedCapture/Corrected.hdr"

# get files info from hdr file
samples, bands, lines, wavelength = extract_envi_info(hdr_path)

# load data and crop in necessary
raw_data = load_raw_hyperspectral_data(raw_path, (600, 35, 200, 395)) # crop(start x, start y, end x, end y)

# manual extraction and plot
A = 5 # the lenght in pixels of the sice of the area
pixel_coordinates = [ (50, 160), (105, 260)] # endmember coordinates
EM_spectra = em_extraction(raw_data, 50, pixel_coordinates, wavelength,A)

# N-findr extraction using pysptools module and plot
EM_NFINDR = eea.NFINDR().extract(M=raw_data, q=2, transform=None, maxit=None, normalize=True, mask=None)
plot_spectra(EM_spectra, EM_NFINDR, wavelength)

# PPI extraction using pysptools module and plot
EM_PPI = eea.PPI().extract(M=raw_data, q=2, numSkewers=1000, normalize=True, mask=None)
plot_spectra(EM_spectra, EM_PPI, wavelength)

# K-means extraction and plot
EM_kmeans, EM_kmeans_n = perform_kmeans(raw_data, 2)
plot_spectra(EM_spectra, EM_kmeans_n, wavelength)

# plot spectral signature map
plot_abundance_map(raw_data, EM_kmeans, ['wheat', 'cadetblue'], bands)
