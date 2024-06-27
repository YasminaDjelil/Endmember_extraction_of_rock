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
import os
import inspect
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import re
from osgeo import gdal
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.decomposition import NMF
import numpy as np
from scipy.signal import savgol_filter
import pysptools.eea as eea
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import euclidean_distances

# necessary func
def extract_wavelengths_from_header(header_file_path):
    with open(header_file_path, 'r') as file:
        content = file.read()
    wavelengths = re.findall(r'\d+\.\d+', content)
    wavelengths = [float(wavelength) for wavelength in wavelengths]
    return wavelengths

def extract_envi_info_from_file(file_path):
    # Extract (samples, lines, bands)
    sample_pattern = re.compile(r'samples\s*=\s*(\d+)', re.IGNORECASE)
    band_pattern = re.compile(r'bands\s*=\s*(\d+)', re.IGNORECASE)
    line_pattern = re.compile(r'lines\s*=\s*(\d+)', re.IGNORECASE)

    with open(file_path, 'r') as file:
        header_text = file.read()

    samples_match = sample_pattern.search(header_text)
    bands_match = band_pattern.search(header_text)
    lines_match = line_pattern.search(header_text)

    samples = int(samples_match.group(1)) if samples_match else None
    bands = int(bands_match.group(1)) if bands_match else None
    lines = int(lines_match.group(1)) if lines_match else None
    # wavelenght
    wavelengths = re.findall(r'\d+\.\d+', header_text)
    wavelengths = [float(wavelength) for wavelength in wavelengths]


    return samples, bands, lines, wavelengths

def load_raw_hyperspectral_data(raw_path, hdr_path, wavelengths, bands, lines, samples, crop_region):

    raw_data = np.empty((bands, lines, samples), dtype=np.uint16)

    # Open the file using gdal
    raw_dataset = gdal.Open(raw_path)

    for i in range(bands):
        band_data = raw_dataset.GetRasterBand(i+1).ReadAsArray()
        raw_data[i, :, :] = band_data

    raw_data = np.transpose(raw_data, (1, 2, 0))
    # wavelenght info
    print("start wavelength =", wavelengths[0])
    print("end wavelength =", wavelengths[-1])
    print("step wavelength =", wavelengths[1]-wavelengths[0])

    # cropping
    if crop_region:
        x_s, y_s, x_e, y_e = crop_region
        cropped_data = raw_data[y_s:y_e, x_s:x_e, :]
        return cropped_data
    else:
        # Reshape
        raw_data = raw_data.reshape(lines, samples, bands)
        return raw_data
      
def plot_data_band(wavelengths, data, Band_index):

  band_data = data[:, :, Band_index]

  # Plot
  plt.figure(figsize=(10, 8))
  plt.imshow(band_data, cmap='gist_gray')
  plt.colorbar(label='Pixel Intensity')
  plt.title('Data from Band {} (Wavelength: {} nm)'.format(Band_index + 1, wavelengths[Band_index]))


  plt.show()

def EM_extraction(header_path, data, Band_index, pixel_coordinates, wavelengths, A):
    band_data = data[:, :, Band_index]

    # Data preview
    plt.figure(figsize=(10, 8))
    plt.imshow(band_data, cmap='gist_gray')
    plt.colorbar(label='Pixel Intensity')
    plt.title('Data from Band {} (Wavelength: {} nm)'.format(Band_index + 1, wavelengths[Band_index]))

    for coord in pixel_coordinates:
        x, y = coord
        plt.scatter(x, y, marker='x', label='Pixel Coordinates')

        x_start, x_end = max(0, int(x - (A//2))), min(data.shape[1], int(x + (A//2)))
        y_start, y_end = max(0, int(y - (A//2))), min(data.shape[0], int(y + (A//2)))
        rect = patches.Rectangle((x_start - 0.5, y_start - 0.5), x_end - x_start, y_end - y_start,
                                  linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), shadow=True, ncol=len(pixel_coordinates)//2)
    plt.show()

    spectra_array = []
    spectra_array_n = []
    for coord in pixel_coordinates:
        x, y = coord

        x_start, x_end = max(0, int(x - (A//2))), min(data.shape[1], int(x + (A//2)))
        y_start, y_end = max(0, int(y - (A//2))), min(data.shape[0], int(y + (A//2)))
        spectra_square = data[y_start:y_end, x_start:x_end]

        spectrum = np.mean(spectra_square, axis=(0, 1))
        # Normalize
        mean_spectrum = (spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))
        spectra_array_n.append(mean_spectrum)
        spectra_array.append(spectrum)

    # Plot the spectra
    plt.figure(figsize=(8, 6))
    plt.title('Spectra of Specified Pixels')
    for i, coord in enumerate(pixel_coordinates):
        x, y = coord
        plt.plot(wavelengths, spectra_array[i], label=f'Pixel {x}, {y} Spectrum')

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend(loc='lower center', bbox_to_anchor=(0.9, -0.2), shadow=True, ncol=len(pixel_coordinates)//2)
    plt.show()

    return np.array(spectra_array_n), np.array(spectra_array)

