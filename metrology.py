import numpy as np
import matplotlib.pyplot as plt
import utilities.imaging.man as man
import utilities.imaging.fitting as fit
import scipy.ndimage as nd
from linecache import getline
import astropy.io.fits as pyfits
import pdb

def readCylScript(fn,rotate=np.linspace(.75,1.5,50),interp=None):
    """
    Load in data from 4D measurement of cylindrical mirror.
    File is assumed to have been saved with Ryan's scripting function.
    Scale to microns, remove misalignments,
    strip NaNs.
    If rotate is set to an array of angles, the rotation angle
    which minimizes the number of NaNs in the image after
    stripping perimeter nans is selected.
    Distortion is bump positive looking at concave surface.
    Imshow will present distortion in proper orientation as if
    viewing the concave surface.
    """
    #Read in values from header
    f = open(fn+'.hdr','r')
    l = f.readlines()
    f.close()
    #Wavelength should never change
    wave = float(l[0].split()[0])*.001 #in microns
    #Ensure wedge factor is 0.5
    wedge = float(l[1])
    if wedge!=0.5:
        print 'Wedge factor is ' + str(wedge)
        pdb.set_trace()
    #Get pixel scale size
    dx = float(l[-1])

    #Remove NaNs and rescale
    d = np.fromfile(fn+'.bin',dtype=np.float32)
    try:
        d = d.reshape((1002,981))
    except:
        d = d.reshape((1003,982))
    d[d>1e10] = np.nan
    d = man.stripnans(d)
    d = d *wave
    d = d - np.nanmean(d)

    #Remove cylindrical misalignment terms
    d = d - fit.fitCylMisalign(d)[0]

    #Rotate out CGH roll misalignment?
    if rotate is not None:
        b = [np.sum(np.isnan(\
            man.stripnans(\
                nd.rotate(d,a,order=1,cval=np.nan)))) for a in rotate]
        d = man.stripnans(\
            nd.rotate(d,rotate[np.argmin(b)],order=1,cval=np.nan))

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx

def readCyl4D(fn,rotate=np.linspace(.75,1.5,50),interp=None):
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
    #Get xpix value in mm
    l = getline(fn,9)
    dx = float(l.split()[1])*1000.
    
    #Remove NaNs and rescale
    d = np.genfromtxt(fn,skip_header=12,delimiter=',')
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)

    #Remove cylindrical misalignment terms
    d = d - fit.fitCylMisalign(d)[0]

    #Rotate out CGH roll misalignment?
    if rotate is not None:
        b = [np.sum(np.isnan(\
            man.stripnans(\
                nd.rotate(d,a,order=1,cval=np.nan)))) for a in rotate]
        d = man.stripnans(\
            nd.rotate(d,rotate[np.argmin(b)],order=1,cval=np.nan))

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx

def readFlatScript(fn,interp=None):
    """
    Load in data from 4D measurement of flat mirror.
    File is assumed to have been saved with Ryan's scripting function.
    Scale to microns, remove misalignments,
    strip NaNs.
    Distortion is bump positive looking at surface from 4D.
    Imshow will present distortion in proper orientation as if
    viewing the surface.
    """
    #Read in values from header
    f = open(fn+'.hdr','r')
    l = f.readlines()
    f.close()
    #Wavelength should never change
    wave = float(l[0].split()[0])*.001 #in microns
    #Ensure wedge factor is 0.5
    wedge = float(l[1])
    if wedge!=0.5:
        print 'Wedge factor is ' + str(wedge)
        pdb.set_trace()
    #Get pixel scale size
    dx = float(l[-1])

    #Remove NaNs and rescale
    d = np.fromfile(fn+'.bin',dtype=np.float32)
    try:
        d = d.reshape((1002,981))
    except:
        d = d.reshape((1003,982))
    d[d>1e10] = np.nan
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)
    d = np.fliplr(d)

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx

def readFlat4D(fn,interp=None):
    """
    Load in data from 4D measurement of flat mirror.
    Scale to microns, remove misalignments,
    strip NaNs.
    Distortion is bump positive looking at surface from 4D.
    Imshow will present distortion in proper orientation as if
    viewing the surface.
    """
    #Get xpix value in mm
    l = getline(fn,9)
    dx = float(l.split()[1])*1000.
    
    #Remove NaNs and rescale
    d = np.genfromtxt(fn,skip_header=12,delimiter=',')
    d = man.stripnans(d)
    d = d *.6328
    d = d - np.nanmean(d)
    d = np.fliplr(d)

    #Interpolate over NaNs
    if interp is not None:
        d = man.nearestNaN(d,method=interp)

    return d,dx

def write4DFits(filename,img,dx,dx2=None):
    """
    Write processed 4D data into a FITS file.
    Axial pixel size is given by dx.
    Azimuthal pixel size is given by dx2 - default to none
    """
    hdr = pyfits.Header()
    hdr['DX'] = dx
    hdu = pyfits.PrimaryHDU(data=img,header=hdr)
    hdu.writeto(filename,clobber=True)
    return

def read4DFits(filename):
    """
    Write FITS file of processed 4D data.
    Returns img,dx in list
    """
    dx = pyfits.getval(filename,'DX')
    img = pyfits.getdata(filename)
    return [img,dx]
