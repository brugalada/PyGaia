#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 18:44:59 2017
Updated on Mon Apr 12 -------- 2021

@author: pramos

Usufull functions for UVW velocity-space treatments
"""
import numpy as np
from .vectorastrometry import *
from .coordinates import *

#constants
# Obliquity of the Ecliptic (arcsec)
_obliquityOfEcliptic = np.deg2rad(84381.41100 / 3600.0)

# Galactic pole in ICRS coordinates (see Hipparcos Explanatory Vol 1 section 1.5, and Murray, 1983,
# section 10.2)
_alphaGalPole = np.deg2rad(192.85948)
_deltaGalPole = np.deg2rad(27.12825)
# The galactic longitude of the ascending node of the galactic plane on the equator of ICRS (see
# Hipparcos Explanatory Vol 1 section 1.5, and Murray, 1983, section 10.2)
_omega = np.deg2rad(32.93192)

_incl=np.deg2rad(62.87124882)    #inclination of galactic plane
_alom=np.deg2rad(282.8594813)    #RA of the equatorial node
#_omega : lom=np.deg2rad(32.93680516)     #longitude gal of Ascending node of the galactic equator
#_deltaGalPole deo=np.deg2rad(27.12825111)     #dec NGP 
#_alphaGalPole alpm=np.deg2rad(192.8594813)    #RA NGP
_theta=np.deg2rad(122.93680516)  #londitude NCP
_const=4.7404705                 #conversion factor km·yr/s


def Jacob_ICRS_to_GAL(ra,dec,parallax,pmra,pmdec,radial_velocity,_fast = True):
    """
    Returns the Jacobian of the transformation from ICRS to GALACTIC.
    
    If '_fast' is True, then the Jacobian is reduced to a 4x4 matrix to speed up calculations (positional errors are ignored).
    
    Parameters
    ----------

    ra : float
        Right ascention (radians).
    dec : float
        Declination (radians).
    parallax : float
        parallax.
    pmra : float
        proper motion in right ascention.
    pmdec: float
        proper motion in declination.
    radial_velocity: float
        velocity along the line of sight.
    _fast: bool, optional
        Ignore positional errors.

    Returns
    -------

    Jacobian of the transformation ICRS - GALACTIC (6x6 or 4x4 matrix)
    """
    l,b=CoordinateTransformation(Transformations.ICRS2GAL).transform_sky_coordinates(ra,dec)

    cd=np.cos(dec);sd=np.sin(dec);tand=np.tan(dec)
    cb=np.cos(b); sb=np.sin(b);tanb=np.tan(b)
    cl=np.cos(l); sl=np.sin(l)
    cdeo=np.cos(_deltaGalPole); sdeo=np.sin(_deltaGalPole)
    cincl=np.cos(_incl); sincl=np.sin(_incl)
    calom=np.cos(ra-_alom); salom=np.sin(ra-_alom); talom=np.tan(ra-_alom)
    clt=np.cos(l-_theta); slt=np.sin(l-_theta)
    
    # initialise
    if _fast:
        jac1=np.zeros([4,4])
        jac2=np.zeros([4,4])
        _offset = 2
    else:
        jac1=np.zeros([6,6])
        jac2=np.zeros([6,6])
        _offset = 0
    
    #from equatorial to galactic coordinates
    
    #l-alpha
    if not _fast:
        jac1[0,0]=((cincl/(calom)**2+sincl/calom*talom*tand)/(1+(cincl*talom+sincl/calom*tand)**2))
    #l-delta
    if not _fast:
        jac1[0,1]=(sincl/(calom*(cd)**2)/(1+(cincl*talom+sincl/calom*tand)**2))
    #b-alpha
    if not _fast:
        jac1[1,0]=(-((calom*cd*sincl)/np.sqrt(1-(cincl*sd-cd*salom*sincl)**2)))
    
    #b-delta
    if not _fast:
        jac1[1,1]=((cd*cincl+salom*sd*sincl)/np.sqrt(1-(cincl*sd-cd*salom*sincl)**2))
        
    #parallax-parallax
    jac1[2-_offset,2-_offset]=1
    
    #mua - mua
    jac1[3-_offset,3-_offset]=1
    
    #mud - mud
    jac1[4-_offset,4-_offset]=1
    
    #vrad - vrad
    jac1[5-_offset,5-_offset]=1
    
    """"""""""""""""""
    #from mua/mub to mul/mub
    
    #l-l
    if not _fast:
        jac2[0,0]=1
    #b-b
    if not _fast:
        jac2[1,1]=1
    #par-par
    jac2[2-_offset,2-_offset]=1
    
    #mul-l
    if not _fast:
        jac2[3,0]=-((pmdec*cdeo*clt)/(1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5))-(pmra*cdeo*( \
        cb*cdeo*clt+sb*sdeo)*(sdeo-sb*(cb*cdeo*clt+sb*sdeo))*slt)/(1-(cb*cdeo*clt+sb*sdeo)**2)**( \
        1.5)+(pmra*cdeo*sb*slt)/(1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5)+(pmdec*cb*cdeo**2*(cb*cdeo*clt+sb*sdeo)*slt*slt)/( \
    1-(cb*cdeo*clt+sb*sdeo)**2)**(1.5)
    
    #ml-b
    if not _fast:
        jac2[3,1]=(pmra/(cb)*(-(cdeo*clt*sb)+cb*sdeo)*(cb*cdeo*clt+sb*sdeo)*(sdeo-sb*(cb*cdeo*clt+sb*sdeo)))/( \
        1-(cb*cdeo*clt+sb*sdeo)**2)**(1.5)+(pmra/(cb)*(-(sb*(-(cdeo*clt*sb)+cb*sdeo))-cb*(cb*cdeo*clt+sb*sdeo)))/ \
    (1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5)-(pmdec*cdeo*(-(cdeo*clt*sb)+cb*sdeo)*(cb*cdeo*clt+sb*sdeo)*slt)/( \
        1-(cb*cdeo*clt+sb*sdeo)**2)**(1.5)+(pmra/(cb)*(sdeo-sb*(cb*cdeo*clt+sb*sdeo))*tanb)/(1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5)
    
    #ml-mua
    jac2[3-_offset,3-_offset]=(1/(cb)*(sdeo-sb*(cb*cdeo*clt+sb*sdeo)))/(1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5)
    
    #ml-mud
    jac2[3-_offset,4-_offset]=-((cdeo*slt)/(1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5))
    
    
    #mb-l
    if not _fast:
        jac2[4,0]=(pmra*cdeo*clt)/(1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5)-(pmdec*cdeo*(cb*cdeo*clt+sb*sdeo)*(sdeo-sb*( \
        cb*cdeo*clt+sb*sdeo))*slt)/(1-(cb*cdeo*clt+sb*sdeo)**2)**(1.5)+(pmdec*cdeo*sb*slt)/(1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5)-( \
        pmra*cb*cdeo**2*(cb*cdeo*clt+sb*sdeo)*slt*slt)/(1-(cb*cdeo*clt+sb*sdeo)**2)**(1.5)
    
    #mb-b
    if not _fast:
        jac2[4,1]=(pmdec/(cb)*(-(cdeo*clt*sb)+cb*sdeo)*(cb*cdeo*clt+sb*sdeo)*(sdeo-sb*(cb*cdeo*clt+sb*sdeo)))/ \
    (1-(cb*cdeo*clt+sb*sdeo)**2)**(1.5)+(pmdec/(cb)*(-(sb*(-(cdeo*clt*sb)+cb*sdeo))-cb*(cb*cdeo*clt+sb*sdeo)))/(1-( \
        cb*cdeo*clt+sb*sdeo)**2)**(0.5)+(pmra*cdeo*(-(cdeo*clt*sb)+cb*sdeo)*(cb*cdeo*clt+sb*sdeo)*slt)/( \
        1-(cb*cdeo*clt+sb*sdeo)**2)**(1.5)+(pmdec/(cb)*(sdeo-sb*(cb*cdeo*clt+sb*sdeo))*tanb)/(1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5)
    
    #mb-mua
    jac2[4-_offset,3-_offset]=(cdeo*slt)/(1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5)
         
    #mb-mud
    jac2[4-_offset,4-_offset]=(1/(cb)*(sdeo-sb*(cb*cdeo*clt+sb*sdeo)))/(1-(cb*cdeo*clt+sb*sdeo)**2)**(0.5)
    
    #vrad-vrad
    jac2[5-_offset,5-_offset]=1
    
    return jac2@jac1


def Jacob_GAL_to_phase_space(l,b,parallax,pml,pmb,radial_velocity,_fast = True):
    """
    Returns the Jacobian of the transformation from GALACTIC to Phase-space.
    The phase space coordinates are assumed to represent barycentric (i.e. centred on the Sun) positions and velocities.
    
    WARNING! The code assumes that the distance is dist = 1/parallax. If a custom distance is used, then simply provide 1/distance as parallax.
    
    If '_fast' is True, then the Jacobian is reduced to a 4x4 matrix to speed up calculations (positional errors are ignored). Else, it considers that the first component of the covariance matrix corresponds to the transformation parallax -> distance.
       
    
    Parameters
    ----------

    l : float
        Galactic longitude (radians).
    b : float
        Galactic latitude (radians).
    parallax : float
        parallax.
    pml : float
        proper motion in Galactic longitude.
    pmb: float
        proper motion in Galactic latitude.
    radial_velocity: float
        velocity along the line of sight.
    _fast: bool, optional
        Ignore positional errors.

    Returns
    -------

    Jacobian of the transformation GALACTIC - Phase-space (6x6 or 4x4 matrix)
    """
    cb=np.cos(b); sb=np.sin(b);tanb=np.tan(b)
    cl=np.cos(l); sl=np.sin(l)
    
    # initialise
    if _fast:
        jac3=np.zeros([4,4])
        jac4=np.zeros([4,4])
        _offset = 2
    else:
        jac3=np.zeros([6,6])
        jac4=np.zeros([6,6])
        _offset = 0
    
    
    # from pml-pmb to uvw
    
    #l-l
    if not _fast:
        jac3[0,0]=1
    #b-b
    if not _fast:
        jac3[1,1]=1
    #par-par
    jac3[2-_offset,2-_offset]=1
    
    #U-l
    if not _fast:
        jac3[3,0]=-((_const*pml*cl)/parallax)-radial_velocity*cb*sl+(_const*pmb*sb*sl)/parallax
    #U-b
    if not _fast:
        jac3[3,1]=-((_const*pmb*cb*cl)/parallax)-radial_velocity*cl*sb
    
    #U-par
    jac3[3-_offset,2-_offset]=(_const*pmb*cl*sb)/parallax**2+(_const*pml*sl)/parallax**2
    
    #U-ml
    jac3[3-_offset,3-_offset]=-((_const*sl)/parallax)
    
    #U-mb
    jac3[3-_offset,4-_offset]=-((_const*cl*sb)/parallax)
    
    #U-vrad
    jac3[3-_offset,5-_offset]=cb*cl
    
    
    #V-l
    if not _fast:
        jac3[4,0]=radial_velocity*cb*cl-(_const*pmb*cl*sb)/parallax-(_const*pml*sl)/parallax
         
    #V-b
    if not _fast:
        jac3[4,1]=-((_const*pmb*cb*sl)/parallax)-radial_velocity*sb*sl
    
    #V-par
    jac3[4-_offset,2-_offset]=-((_const*pml*cl)/parallax**2)+(_const*pmb*sb*sl)/parallax**2
    
    #V-ml
    jac3[4-_offset,3-_offset]=(_const*cl)/parallax
    
    #V-mb
    jac3[4-_offset,4-_offset]=-((_const*sb*sl)/parallax)
    
    #V-vrad
    jac3[4-_offset,5-_offset]=cb*sl
    
    
    #W-l
    if not _fast:
        jac3[5,0]=0
    #W-b
    if not _fast:
        jac3[5,1]=radial_velocity*cb-(_const*pmb*sb)/parallax
    #W-par
    jac3[5-_offset,2-_offset]=-((_const*pmb*cb)/parallax**2)
    #W-ml
    jac3[5-_offset,3-_offset]=0
    #W-mb
    jac3[5-_offset,4-_offset]=(_const*cb)/parallax
    #W-vrad
    jac3[5-_offset,5-_offset]=sb
    
    
    # l,b,plx to x,y,z
    
    if not _fast:
        #l-x
        jac4[0,0]=-sl*cb/parallax
        #l-y
        jac4[1,0]=cl*cb/parallax
        #b-x
        jac4[0,1]=-cl*sb/parallax
        #b-y
        jac4[1,1]=-sl*sb/parallax
        #b-z
        jac4[2,1]=cb/parallax
        #plx-x
        jac4[0,2]=-cl*cb/parallax**2
        #plx-y
        jac4[1,2]=-sl*cb/parallax**2
        #plx-z
        jac4[2,2]=-sb/parallax**2
    else:
        #parallax-distance
        jac4[0,0] = -1/parallax**2
    
    jac4[3-_offset,3-_offset]=1
    jac4[4-_offset,4-_offset]=1
    jac4[5-_offset,5-_offset]=1
    
    return jac4@jac3


def Jacob_phase_space_to_galcen(x,y,z,vx,vy,vz,_fast = True,Dsun=8.178,Zsun=0.0208):
    """
    Returns the Jacobian of the transformation from Phase-space to Galactocentric cylindrical.
    The phase space coordinates are assumed to represent barycentric (i.e. centred on the Sun) positions and velocities.
    
        
    If '_fast' is True, then the Jacobian is reduced to a 4x4 matrix to speed up calculations (positional errors are ignored). Else, it considers that the first component of the covariance matrix is the error in distance and is left untransformed.
       
    
    Parameters
    ----------

    x - The x component of the barycentric position vector (in kpc).
    y - The y component of the barycentric position vector (in kpc).
    z - The z component of the barycentric position vector (in kpc).
    vx - The x component of the barycentric velocity vector (in km/s).
    vy - The y component of the barycentric velocity vector (in km/s).
    vz - The z component of the barycentric velocity vector (in km/s).
    Dsun (optional) - distance from the Sun to Sgr A* (kpc). Default: 8.178 (Gravity Collaboration et al. 2019)
    Zsun (optional) - distance from the Sun to the galactic midplane (kpc). Default: 0.0208 (Bennett & Bovy 2019)
    _fast: bool, optional
        Ignore positional errors.

    Returns
    -------

    Jacobian of the transformation Phase-space - Galactocentric cartesian (6x6 or 4x4 matrix)
    """

    #constants
    Xsun=np.sqrt(Dsun**2.-Zsun**2.)
    costheta, sintheta= Xsun/Dsun, -Zsun/Dsun
    
    # initialise
    if _fast:
        jac5=np.zeros([4,4])
        _offset = 2
    else:
        jac5=np.zeros([6,6])
        _offset = 0
    
    
    # rotate cartesian frame to align with Galactic plane
    
    if not _fast:
        #x-x'
        jac5[0,0]=costheta
        #z-x'
        jac5[0,2]=sintheta
        #x-z'
        jac5[2,0]=-sintheta
        #z-z'
        jac5[2,2]=costheta
        #y-y'
        jac5[1,1]=1
    else:
        #distance-distance
        jac5[0,0] = 1
       
    #vx-vx'
    jac5[3-_offset,3-_offset]=costheta
    
    #vz-vx'
    jac5[3-_offset,5-_offset]=-sintheta
    
    #vx-vz'
    jac5[5-_offset,3-_offset]=-sintheta
    
    #vz-vz'
    jac5[5-_offset,5-_offset]=costheta
    
    #vy-vy'
    jac5[4-_offset,4-_offset]=1
    
    return jac5


def Jacob_galcen_to_cyl(x,y,z,vx,vy,vz,_fast = True):
    """
    Returns the Jacobian of the transformation from Galactocentric cartesian to Galactocentric cylindrical.
    
        
    If '_fast' is True, then the Jacobian is reduced to a 4x4 matrix to speed up calculations (positional errors are ignored). Else, it considers that the first component of the covariance matrix is the error in distance and is left untransformed.
       
    
    Parameters
    ----------

    x - The x component of the galactocentric position vector (in kpc).
    y - The y component of the galactocentric position vector (in kpc).
    z - The z component of the galactocentric position vector (in kpc).
    vx - The x component of the galactocentric velocity vector (in km/s).
    vy - The y component of the galactocentric velocity vector (in km/s).
    vz - The z component of the galactocentric velocity vector (in km/s).
    _fast: bool, optional
        Ignore positional errors.

    Returns
    -------

    Jacobian of the transformation Galactocentric cartesian - Galactocentric cylindrical (6x6 or 4x4 matrix)
    """
    #r,phi,z = cartesian_to_cylindrical(x,y,z,_galcentric=True)
    r    = np.sqrt(x**2+y**2)
    cphi = x/r
    sphi = y/r
    
    # initialise
    if _fast:
        jac6=np.zeros([4,4])
        _offset = 2
    else:
        jac6=np.zeros([6,6])
        _offset = 0

    # x,y,z,vx,vy,vz to r,phi,z,vr,vphi,vz
    if not _fast:
        #x'-r
        jac6[0,0]= -cphi
        #y'-r
        jac6[0,1]= -sphi
        #x'-phi
        jac6[1,0]= y/(x**2+y**2)
        #y'-phi
        jac6[1,1]= -x/(x**2+y**2)
        #z-z'
        jac6[2,2]= 1
        
    else:
        #distance-distance
        jac6[0,0] = 1
    
    if not _fast:
        #x'-vr
        jac6[3,0]= vx*sphi**2/r - vy*cphi*sphi/r

        #y'-vr
        jac6[3,1]= -vx*cphi*sphi/r+vy*cphi**2/r
    
    #vx'-vr
    jac6[3-_offset,3-_offset]= +cphi
    
    #vy'-vr
    jac6[3-_offset,4-_offset]= +sphi
    
    if not _fast:
        #x'-vphi
        jac6[4,0]= +vx*cphi*sphi/r + vy*sphi**2/r

        #y'-vphi
        jac6[4,1]= -vx*cphi**2/r - vy*cphi*sphi/r
    
    #vx'-vphi
    jac6[4-_offset,3-_offset]= -sphi
    
    #vy'-vphi
    jac6[4-_offset,4-_offset]= +cphi
    
    #vz-vz'
    jac6[5-_offset,5-_offset]= 1
    
    
    return jac6
