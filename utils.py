def nonzero(x):
    """
    If given 0 then this returns an extremely tiny, but non-zero, positive value. Otherwise the
    given value is returned. The value x must be a scalar and cannot be an array.
    """
    from numpy import finfo
    return finfo(float).eps if x == 0 else x

def psf2otf(psf, shape):
    """
    Convert a PSF to an OTF. This is essentially np.fft.fft2(psf, shape) except it also includes a
    minor shift of the data so that the center of the PSF is at (0,0) before the Fourier transform
    is computed but after padding.
    """
    from numpy import pad, roll
    from numpy.fft import fft2
    psf_shape = psf.shape
    psf = pad(psf, ((0, shape[0] - psf_shape[0]), (0, shape[1] - psf_shape[1])), 'constant')
    psf = roll(psf, (-(psf_shape[0]//2), -(psf_shape[1]//2)), axis=(0,1)) # shift PSF so center is at (0,0)
    return fft2(psf)

def otf2psf(otf, shape):
    """
    Convert an OTF to a PSF. This is essentially np.fft.ifft2(otf, shape) except it also includes a
    minor shift of the data so that the center of the PSF is moved back to the middle after the
    inverse Fourier transform is computed but before cropping.
    """
    from numpy import roll
    from numpy.fft import ifft2
    psf = ifft2(otf)
    psf = roll(psf, (shape[0]//2, shape[1]//2), axis=(0,1)) # shift PSF so center is in middle
    return psf[:shape[0], :shape[1]]

def __x2y2(w, h):
    """Creates the mesh grid of x^2 + y^2 from -w/2 to w/2 and -h/2 to h/2"""
    from numpy import ogrid
    x,y = ogrid[-(w//2):((w+1)//2), -(h//2):((h+1)//2)]
    return x*x + y*y
    
def ideal_low_pass(w, h, D):
    """
    Creates a Fourier-space ideal low-pass filter of the given width and height with the cutoff D.
    """
    return (__x2y2(w,h)<=(D*D)).astype(float)
    
def ideal_high_pass(w, h, D):
    """
    Creates a Fourier-space ideal high-pass filter of the given width and height with the cutoff D.
    """
    return (__x2y2(w,h)>(D*D)).astype(float)

def butterworth_low_pass(w, h, D, n):
    """
    Creates a Fourier-space Butterworth low-pass filter of the given width and height with the
    cutoff D and order n.
    """
    return 1 / (1 + (__x2y2(w,h)/(D*D))**n)
    
def butterworth_high_pass(w, h, D, n):
    """
    Creates a Fourier-space Butterworth high-pass filter of the given width and height with the
    cutoff D and order n.
    """
    return 1 - butterworth_low_pass(w, h, D, n)

def gaussian(w, h, sigma, normed=False):
    """
    Creates a Gaussian centered in a w x h image with the given standard deviation sigma. By
    default this has a peak of 1. If normed is True then this is normalized so it sums to 1.
    """
    from numpy import exp
    g = exp(-__x2y2(w,h)/(sigma*sigma))
    if normed: g /= g.sum()
    return g

def fftshow(arr, mode='mag', log_scale=True, eliminate_dc=True, plot=True):
    """
    Shows a 2D Fourier transform using one of the following modes:
      * "mag" - the complex magnitude (default)
      * "color" - map complex numbers onto a color wheel using HSV color model
      
    For color mapping the complex angle is the hue (red is pos. real, cyan is
    neg. real, yellow/green is pos. imaginary, and purple is neg. imaginary),
    the saturation is 1, and the values are the complex magnitude; some hue
    adjustments are made to neighboring angles that are 180 from each other.
    
    By default the DC component of the image is eliminated and the magnitude
    data is log-scaled to make them displayable relative to each other. Set
    `log_scale` and `eliminate_dc` to False to turn these off.
    
    Instead of plotting the image the generated image can be returned so that
    it can be saved or otherwise manipulated.
    """
    from numpy import median, log, unravel_index, arctan2, pi, ones, dstack
    
    # Calculate the complex magnitude
    mag = abs(arr)
    
    # Find the DC component
    dc_i,dc_j = unravel_index(mag.argmax(), mag.shape)
    # Eliminate the DC component
    if eliminate_dc:
        mag[dc_i,dc_j] = median(mag[max(0,dc_i-1):dc_i+2,max(0,dc_j-1):dc_j+2])
        
    # Logarithm scaling
    if log_scale: mag = log(mag+1)
    
    # Map image data
    if mode == 'mag': im = mag # Just magnitude
    elif mode == 'color':
        # Color mapping using HSV
        # Hue is based on the angle of the complex number
        H = arctan2(arr.imag, arr.real) # Calculate the complex angle (-pi to pi)
        H[::2,::2] += pi; H[1::2,1::2] += pi # adjust for cross-hatch pattern of angles
        H /= 2*pi; H %= 1 # convert from -pi to pi range to 0 to 1
        H[dc_i, dc_j] = 0 # correct DC component angle
        
        # The saturation is all 1s
        S = ones(arr.shape)
        
        # The values are the normalized magnitude of the complex number (from 0 to 1)
        mag -= mag.min(); mag /= mag.max(); V = mag

        # Convert HSV to RGB for displaying
        from matplotlib.colors import hsv_to_rgb
        im = hsv_to_rgb(dstack((H, S, V)))
    
    # Unknown mode
    else: raise ValueError('mode must be one of "mag" or "color"')
        
    # Show or return the image
    if plot:
        from matplotlib.pylab import imshow
        imshow(im)
    else: return im
