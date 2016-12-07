import numpy as np
import matplotlib.pyplot as plt
import utilities.imaging.man as man
import utilities.imaging.fitting as fit
import scipy.ndimage as nd

def readCyl4D(fn,rotate=np.linspace(.75,1.25,50),interp=None):
    """
    Load in data from 4D measurement of cylindrical mirror.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Remove NaNs and rescale
    d = np.genfromtxt(fn,skip_header=12,delimiter=',')
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)

    #Remove cylindrical misalignment terms
    d = d - fit.fitCylMisalign(d)[0]

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    #Rotate out CGH roll misalignment?
    if rotate is not None:
        b = [np.sum(np.isnan(\
            man.stripnans(\
                nd.rotate(d,a,order=1,cval=np.nan)))) for a in rotate]
        pdb.set_trace()
        return man.stripnans(\
            nd.rotate(d,rotate[np.argmin(b)],order=1,cval=np.nan))

    return d

def readFlat4D(fn,interp=None):
    """
    Load in data from 4D measurement of flat mirror.
    Scale to microns, remove misalignments,
    strip NaNs.
    Distortion is bump positive looking at surface from 4D.
    Imshow will present distortion in proper orientation as if
    viewing the surface.
    """
    #Remove NaNs and rescale
    d = np.genfromtxt(fn,skip_header=12,delimiter=',')
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)
    d = np.fliplr(d)

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d
