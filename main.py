import numpy as np

def asprop(e, z, λ, d):
    """
    Angular Spectrum method of propagation on unshifted complex field (unitless)
    INPUTS:
        e: [complexe array] amplitude field; e.g.: E*exp(j*phi)
        z: distance to propagation plane; e.g.: 1e3
        λ: [float] wavelength > 0
        d: [tuple] size of meta-cell (nx>0, ny>0); size of pixels
    OUTPUT:
        stack of complex field in propagation plane(s)
    """
    # invert field phase for back-propagation
    if np.sign(z) < 0:
        e = np.conjugate(e)
    
    # compute angular spectrum
    E = np.fft.fftshift(np.fft.fft2(e))
    
    # extract grid parameters
    nx, ny = e.shape
    dx, dy = d
    
    # get k-grid (spatial frequency); real mode:propagating; complex mode: evanescent
    k = 2 * np.pi / λ
    k_x = np.fft.fftshift(np.fft.fftfreq(n=nx, d=dx) * 2 * np.pi)
    k_y = np.fft.fftshift(np.fft.fftfreq(n=ny, d=dy) * 2 * np.pi)
    k_Y, k_X = np.meshgrid(k_y, k_x, indexing='xy')
    kz = np.sqrt(0j + k ** 2 - k_X ** 2 - k_Y ** 2)
    
    # calculate diffraction plane angular spectrum
    Ez = E * np.exp(1j * kz * np.abs(z))
    
    # retrieve diffraction-plane real-space field
    ez = np.fft.ifft2(np.fft.ifftshift(Ez))
    
    return ez[::-1]

if __name == "__main__":
#     e = import the complex aperture here
    f = asprop(e, z, λ, d)
