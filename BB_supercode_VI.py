#V2: Using BB search to identify line channels first.
#V3: Adds outputs to master lists (for use on simobs'd cubes)
#-->For use on simobs galaxies, etc.
#V_I: For creating M0, M1, M2, PV_M, PV_m fits and placing them with CONT in a given file
#Gets W15 1-5, G_C, G_M0, M20_C, M20_M0
#V_II: Moves from central mask to user-provided S/N thresholds
#V_III: Generalized for other (non-ALPINE) cubes
#V_IV: Fixing axes, removing continuum, adding dictionaries
#V_VI: Adding channel maps, generalizing for desk/lap
#VI: Moving from BB1.4 to BB1.5

import sys
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import unumpy as unp
from matplotlib.patches import Ellipse
from __casac__.image import *
from __casac__.regionmanager import *
import os
from uncertainties import unumpy as unp
import FUNCTIONS as fn
from astropy.wcs import WCS
sys.path.append("/Users/garethjones/Documents/")
import matplotlib.cm as cm
from scipy.optimize import curve_fit
import getrms_v3_auto as GRMS
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append("/Users/garethjones/Desktop/scripts/")
import cosmocalc2 as CC
sys.path.append("/usr/local/bin/gnuplot")
import gzip
import shutil
from matplotlib import colors

#BBFOLD='/Users/garethjones/Documents/BBarolo-1.4/';BBCOMMAND='./BBarolo';RINGFILE='ringlog'
BBFOLD='/Users/garethjones/Documents/Bbarolo-1.5/';BBCOMMAND='./BBarolo-1.5';RINGFILE='ringlog'
#BBFOLD='/Users/garethjones/Documents/Bbarolo-1.6/';BBCOMMAND='./BBarolo';RINGFILE='rings_final'

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Constants

cspl=299792.458 #3E+5 isn't precise enough [km/s]
fitsig=True
cfact=2*np.sqrt(2*np.log(2))
normtype='local' #local/azim/none
bigdata=[]
outputfolder='/Users/garethjones/Desktop/BBOUT/'
cosmoparams=[70,0.3,0.7]
upsidedown=False
#Some defaults
vsysestimate=0.
resm0=[-1,-1,-1]
m0box="-1"
xy0=[-1,-1]
XYDIST=25.
SIDE='B'
vrotguess=-1
delv2=-1
makepanels=False
mrradfact=0.8
linw=2
DATA_MASK='DATA' #DATA/MASK
MASKTYPE='SEARCH'
bbxlim=False; bbxlim2=False
unconinc=True
dotwostage=True
panellabels=True #Add labels for first three rows [M0 (Data), etc]?
dogrid=True #Add grid for rows 2 & 3?
addcross=True #Add 5kpc x5kpx cross in row 1?
nicetext=True #Boldface labels and ticks?
m0fact=1.0
addcontours=True

michelesetup=False
if michelesetup:
	panellabels=False
	dogrid=False
	addcross=False
	makepanels=True

#-
#Initialize contour levels (moment zero, channel maps)
maxc=100
lev_1=[2,4,8,16,32,64,128,256,512,1024]
lev_2=list(reversed(lev_1))
lev_both_2=np.zeros(len(lev_1)*2)
for i in range(len(lev_2)):
	lev_both_2[i]=-1*lev_2[i]
	lev_both_2[i+len(lev_1)]=lev_1[i]
#-

#Doesn't quite work yet anyways.
MakeChanMaps=False

#The version of MPL in CASA doesn't have viridis or plasma. Gotta import them.
import colormaps as cmaps

cmap1='gist_heat_r'#cmaps.viridis
cmap2='bwr'#cmaps.plasma
cmap3=cmaps.inferno
cmap4=cmaps.viridis


#https://medialab.github.io/iwanthue/
threecolors=['#749d70','#a76bb2','#b5863e']

if nicetext:
	fs=13; fs2=25
	font = {'family' : 'normal','weight' : 'bold','size' : fs}
	matplotlib.rc('font', **font)
else:
	fs=13; fs2=25
	font = {'family' : 'normal','weight' : 'normal','size' : fs}
	matplotlib.rc('font', **font)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Functions

#Takes in list of beams, returns median beam
def avgbeam(bmlist):
	majL=[]
	minL=[]
	paL=[]
	for i in range(bmlist.shape[0]):
		temp=str(bmlist[i]).split(', ')
		majL.append(float(temp[0].replace('(','')))
		minL.append(float(temp[1]))
		if (float(temp[2])>360.) or (float(temp[2])<-360.):
			print 'BPA UNREASONABLE:',temp[2],float(temp[2])%360.
			paL.append(float(temp[2])%360.)
		else:
			paL.append(float(temp[2]))
	return [np.median(majL)/3600.,np.median(minL)/3600.,np.median(paL)]

#Velocity/frequency conversions
def VtoNU(v,restFreq):
	temp=restFreq/(1+(v/cspl))
	return temp*(1E-9)
def NUtoV(nu,restFreq):
	temp=cspl*((restFreq/nu)-1)
	return temp

#Takes in index of parameter, index of first error.
#Outputs weighted average and uncertainty
def getweighted(ii,j):
		try:
			#Get list of values
			XXList=RUN1[:,ii]
			NRINGS2=len(XXList); XXList_E=np.zeros(NRINGS2); XXList_w=np.zeros(NRINGS2)
			tempweight=0.; XXavg=0.; XXstd=0.; temp2=0.
			#Get errors from ringfile and weight normalization factor
			for i in range(NRINGS2): 
				XXList_E[i]=max(abs(RUN1[i,j]),RUN1[i,j+1])
				tempweight+=(1/XXList_E[i])
			#Get normalized weights, weighted average
			for i in range(NRINGS2):
				XXList_w[i]=(1/XXList_E[i])/tempweight
				XXavg+=XXList[i]*XXList_w[i]
			for i in range(NRINGS2):
				temp2+=((XXList[i]-XXavg)**2)*XXList_w[i]
			XXstd=np.sqrt((NRINGS2/(NRINGS2-1))*temp2/tempweight)
			return XXavg,XXstd
		except IndexError:
			return RUN1[ii],max(abs(RUN1[j]),RUN1[j+1])	

#Takes in index of parameter, index of first error.
#Outputs best linear fit (zero slope) and uncertainty
def flatline(x,a):
	temp=a
	return temp
def getweighted2(ii,j):
		try:
			#Get list of values
			XXList=RUN1[:,ii]
			RadList=RUN1[:,1]
			NRINGS2=len(XXList); XXList_E=np.zeros(NRINGS2); XXList_w=np.zeros(NRINGS2)
			tempweight=0.; XXavg=0.; XXstd=0.; temp2=0.
			#Get errors from ringfile
			for i in range(NRINGS2): 
				XXList_E[i]=max(abs(RUN1[i,j]),RUN1[i,j+1])
			flatguess=np.average(XXList)
			popt, pcov = curve_fit(flatline, RadList, XXList, p0=flatguess, sigma=XXList_E)
			if np.sqrt(pcov[0,0])!=0.:
				T_avg=popt[0]; T_std=np.sqrt(pcov[0,0])
			else:
				T_avg=popt[0]; T_std=max(XXList_E)
			return T_avg,T_std
		except IndexError:
			print 'BAD getweighted2----------'
			return RUN1[ii],max(abs(RUN1[j]),RUN1[j+1])		

def polyn_t(x,a,b):
	temp=(a*x)+b
	return temp

#Splits uarray string into actual array [val,err]
def splituarray(uar):
	uarr=str(uar)
	if 'e' in str(uar):
		part1=uarr.split('+/-')[0].replace('(','')
		part2=uarr.split('+/-')[1].replace(')','')
		part2a=part2.split('e')[0];part2b=part2.split('e')[1]
		P1=float(part1)*(10**float(part2b))
		P2=float(part2a)*(10**float(part2b))
	else:
		P1=uarr.split('+/-')[0].replace('(','')
		P2=uarr.split('+/-')[1].replace(')','')
	return [float(P1),float(P2)]

#Takes in index of parameter, index of first error.
#Outputs average and uncertainty, with no weighting
def getweighted3(ii,j):
	VI=[]; DVI=[]
	try:
		for idk3 in range(len(RUN1[:,ii])):
			VI.append(RUN1[idk3,ii])
			DVI.append(max(abs(RUN1[idk3,j]),RUN1[idk3,j+1]))
		tempvi1=0
		tempdvi1=0
		for idk3 in range(len(RUN1[:,ii])):
			tempvi1+=VI[idk3]/len(RUN1[:,ii])
			tempdvi1+=(DVI[idk3]**2)
		return tempvi1,np.sqrt(tempdvi1/len(RUN1[:,ii]))
	except IndexError: #One ring
		VI.append(RUN1[ii])
		DVI.append(max(abs(RUN1[j]),RUN1[j+1]))
		tempvi1=RUN1[ii]
		tempdvi1=(max(abs(RUN1[j]),RUN1[j+1])**2)
		return tempvi1,np.sqrt(tempdvi1)

#Fixes the CASA 4.7.2 fits header issue
def fixcube(incube,restFreq,outcube):
	ksfile=incube
	hdunew=fits.PrimaryHDU()
	hduold=fits.open(ksfile)
	hdunew.data=np.ones((hduold[0].data.shape[0],hduold[0].data.shape[1]))
	hdunew.header['CDELT1']=hduold[0].header['CDELT1']
	hdunew.header['CDELT2']=hduold[0].header['CDELT2']
	hdunew.header['CRPIX1']=hduold[0].header['CRPIX1']
	hdunew.header['CRPIX2']=hduold[0].header['CRPIX2']
	hdunew.header['CRVAL1']=hduold[0].header['CRVAL1']
	hdunew.header['CRVAL2']=hduold[0].header['CRVAL2']
	hdunew.header['CTYPE1']=hduold[0].header['CTYPE1']
	hdunew.header['CTYPE2']=hduold[0].header['CTYPE2']
	hdunew.header['CUNIT1']=hduold[0].header['CUNIT1']
	hdunew.header['CUNIT2']=hduold[0].header['CUNIT2']
	#hdunew.header['FREQ0']=hduold[0].header['FREQ0']
	try:
		hdunew.header['RESTFRQ']=hduold[0].header['RESTFRQ']
	except KeyError:
		pass
	try:
		hdunew.header['EQUINOX']=hduold[0].header['EQUINOX']
	except KeyError:
		hdunew.header['EQUINOX']=2000.0
	hdunew.header['BMAJ']=hduold[0].header['BMAJ']
	hdunew.header['BMIN']=hduold[0].header['BMIN']
	hdunew.header['BPA']=hduold[0].header['BPA']
	hdunew.header['BUNIT']=hduold[0].header['BUNIT']
	hdunew.header['RADESYS']=hduold[0].header['RADESYS']
	hdunew.data[:,:]=hduold[0].data
	hduold.close()
	hdunew.writeto(outcube,overwrite=True)
  
def fixcube3d(incube,restFreq,outcube):
	ksfile=incube
	hdunew=fits.PrimaryHDU()
	hduold=fits.open(ksfile)	
	if len(hduold[0].data.shape)==4:
		if hduold[0].data.shape[0]==1:
			ZSIZE=hduold[0].data.shape[1]
			YSIZE=hduold[0].data.shape[2]
			XSIZE=hduold[0].data.shape[3]
			if XSIZE!=YSIZE: 
				print 'BAD FIXCUBE3D_1'
			hdunew.data=np.ones((hduold[0].data.shape[1],hduold[0].data.shape[2],hduold[0].data.shape[3]))
			hdunew.header['CDELT1']=hduold[0].header['CDELT1']
			hdunew.header['CDELT2']=hduold[0].header['CDELT2']
			hdunew.header['CDELT3']=hduold[0].header['CDELT3']
			hdunew.header['CRPIX1']=hduold[0].header['CRPIX1']
			hdunew.header['CRPIX2']=hduold[0].header['CRPIX2']
			hdunew.header['CRPIX3']=hduold[0].header['CRPIX3']
			hdunew.header['CRVAL1']=hduold[0].header['CRVAL1']
			hdunew.header['CRVAL2']=hduold[0].header['CRVAL2']
			hdunew.header['CRVAL3']=hduold[0].header['CRVAL3']
			hdunew.header['CTYPE1']=hduold[0].header['CTYPE1']
			hdunew.header['CTYPE2']=hduold[0].header['CTYPE2']
			hdunew.header['CTYPE3']=hduold[0].header['CTYPE3']
			hdunew.header['CUNIT1']=hduold[0].header['CUNIT1']
			hdunew.header['CUNIT2']=hduold[0].header['CUNIT2']
			hdunew.header['CUNIT3']=hduold[0].header['CUNIT3']
		elif hduold[0].data.shape[3]==1:
			ZSIZE=hduold[0].data.shape[0]
			YSIZE=hduold[0].data.shape[1]
			XSIZE=hduold[0].data.shape[2]
			if XSIZE!=YSIZE: 
				print 'BAD FIXCUBE3D_2'
			hdunew.data=np.ones((hduold[0].data.shape[1],hduold[0].data.shape[2],hduold[0].data.shape[3]))
			hdunew.header['CDELT1']=hduold[0].header['CDELT2']
			hdunew.header['CDELT2']=hduold[0].header['CDELT3']
			hdunew.header['CDELT3']=hduold[0].header['CDELT4']
			hdunew.header['CRPIX1']=hduold[0].header['CRPIX2']
			hdunew.header['CRPIX2']=hduold[0].header['CRPIX3']
			hdunew.header['CRPIX3']=hduold[0].header['CRPIX4']
			hdunew.header['CRVAL1']=hduold[0].header['CRVAL2']
			hdunew.header['CRVAL2']=hduold[0].header['CRVAL3']
			hdunew.header['CRVAL3']=hduold[0].header['CRVAL4']
			hdunew.header['CTYPE1']=hduold[0].header['CTYPE2']
			hdunew.header['CTYPE2']=hduold[0].header['CTYPE3']
			hdunew.header['CTYPE3']=hduold[0].header['CTYPE4']
			hdunew.header['CUNIT1']=hduold[0].header['CUNIT2']
			hdunew.header['CUNIT2']=hduold[0].header['CUNIT3']
			hdunew.header['CUNIT3']=hduold[0].header['CUNIT4']
		else:
			print 'BAD FIXCUBE3D_3'
	else:
		hdunew.data=np.ones((hduold[0].data.shape[0],hduold[0].data.shape[1],hduold[0].data.shape[2]))
		hdunew.header['CDELT1']=hduold[0].header['CDELT1']
		hdunew.header['CDELT2']=hduold[0].header['CDELT2']
		hdunew.header['CDELT3']=hduold[0].header['CDELT3']
		hdunew.header['CRPIX1']=hduold[0].header['CRPIX1']
		hdunew.header['CRPIX2']=hduold[0].header['CRPIX2']
		hdunew.header['CRPIX3']=hduold[0].header['CRPIX3']
		hdunew.header['CRVAL1']=hduold[0].header['CRVAL1']
		hdunew.header['CRVAL2']=hduold[0].header['CRVAL2']
		hdunew.header['CRVAL3']=hduold[0].header['CRVAL3']
		hdunew.header['CTYPE1']=hduold[0].header['CTYPE1']
		hdunew.header['CTYPE2']=hduold[0].header['CTYPE2']
		hdunew.header['CTYPE3']=hduold[0].header['CTYPE3']
		hdunew.header['CUNIT1']=hduold[0].header['CUNIT1']
		hdunew.header['CUNIT2']=hduold[0].header['CUNIT2']
		hdunew.header['CUNIT3']=hduold[0].header['CUNIT3']
	if restFreq==999:
		try:
			hdunew.header['RESTFRQ']=hduold[0].header['RESTFRQ']
		except KeyError:
			pass
	else:
		hdunew.header['RESTFRQ']=restFreq
	try:
		hdunew.header['VELREF']=hduold[0].header['VELREF']
	except KeyError:
		hdunew.header['VELREF']=257
	try:
		hdunew.header['EQUINOX']=hduold[0].header['EQUINOX']
	except KeyError:
		hdunew.header['EQUINOX']=2000.0
	try:
		hdunew.header['BMAJ']=hduold[0].header['BMAJ']
		hdunew.header['BMIN']=hduold[0].header['BMIN']
		hdunew.header['BPA']=hduold[0].header['BPA']
		if float(hduold[0].header['BPA'])<-360. or float(hduold[0].header['BPA'])>360.:
			print 'BPA UNREASONABLE:',hduold[0].header['BPA'],float(hduold[0].header['BPA'])%360.
			hdunew.header['BPA']=float(hduold[0].header['BPA'])%360.
	except KeyError:
		hdunew.header['BMAJ']=avgbeam(hduold[1].data)[0]
		hdunew.header['BMIN']=avgbeam(hduold[1].data)[1]
		hdunew.header['BPA']=avgbeam(hduold[1].data)[2]
	hdunew.header['BUNIT']=hduold[0].header['BUNIT']
	try:
		hdunew.header['RADESYS']=hduold[0].header['RADESYS']
	except KeyError:
		hdunew.header['RADESYS']='ICRS'
	try:
		hdunew.header['SPECSYS']=hduold[0].header['SPECSYS']
	except KeyError:
		hdunew.header['SPECSYS']='LSRK'
	try:
		hdunew.header['ALTRVAL']=hduold[0].header['ALTRVAL']
		hdunew.header['ALTRPIX']=hduold[0].header['ALTRPIX']
	except KeyError: 
		pass
	hdunew.data[:,:,:]=hduold[0].data
	hduold.close()
	hdunew.writeto(outcube,overwrite=True)

#Zoom in, add real labels, plot beam, plot rotation axes
def makepretty(wbar,PA,ax,xcen_MP,ycen_MP):
	#Find plot limits
	ax.set_xlim(xcen_MP-1,xcen_MP+1)
	ax.set_ylim(ycen_MP-1,ycen_MP+1)
	#Account for rounding errors
	for uhh in range(len(xlist2)):
		if xlist2[uhh]>0 and xlist2[uhh]<0.1:
			xlist2[uhh]=0.
		if ylist2[uhh]>0 and ylist2[uhh]<0.1:
			ylist2[uhh]=0.
	xlist1=[(1.*xl2/(L_head['cdelt1']*3600.))+xcen_MP for xl2 in xlist2]
	ax.set_xticks(xlist1)
	ax.set_xticklabels(xlist2,rotation=0)
	ylist1=[(1.*yl2/(L_head['cdelt2']*3600.))+ycen_MP for yl2 in ylist2]
	ax.set_yticks(ylist1)
	ax.set_yticklabels(ylist2)
	overhang=0.1
	ax.set_xlim(((zooom+overhang)/(L_head['cdelt1']*3600.))+xcen_MP,((-1*zooom-overhang)/(L_head['cdelt1']*3600.))+xcen_MP)
	ax.set_ylim(((-1*zooom-overhang)/(L_head['cdelt2']*3600.))+ycen_MP,((zooom+overhang)/(L_head['cdelt2']*3600.))+ycen_MP)
	
	#Plot beam
	xb=0.15*(ax.get_xlim()[1]-ax.get_xlim()[0])+ax.get_xlim()[0]
	yb=ax.get_ylim()[0]+0.15*(ax.get_ylim()[1]-ax.get_ylim()[0])
	ell1=Ellipse(xy=[xb,yb],height=L_BMAJ_c*cfact,width=L_BMIN_c*cfact,angle=BPA,fill=True,facecolor='w',hatch='/')
	ax.add_artist(ell1)
	if wbar:
		#Plot rotation axes
		truecenter=[xcen_MP,ycen_MP]
		PA_r=(PA)*(np.pi/180.)
		ax_M_x=[truecenter[0]-(length/2.)*np.sin(PA_r),truecenter[0],truecenter[0]+(length/2.)*np.sin(PA_r)]
		ax_M_y=[truecenter[1]+(length/2.)*np.cos(PA_r),truecenter[1],truecenter[1]-(length/2.)*np.cos(PA_r)]
		ax.plot(ax_M_x,ax_M_y,color='k',linewidth=2)
		PA_m_r=(PA+90.)*(np.pi/180.)
		ax_m_x=[truecenter[0]-(length/2.)*np.sin(PA_m_r),truecenter[0],truecenter[0]+(length/2.)*np.sin(PA_m_r)]
		ax_m_y=[truecenter[1]+(length/2.)*np.cos(PA_m_r),truecenter[1],truecenter[1]-(length/2.)*np.cos(PA_m_r)]
		ax.plot(ax_m_x,ax_m_y,color='k',linestyle='dashed',linewidth=2)

#For chan maps
def makepretty2(wbar,PA,ax,xcen_MP,ycen_MP,xon,yon):
	#Find plot limits
	ax.set_xlim(xcen_MP-1,xcen_MP+1)
	ax.set_ylim(ycen_MP-1,ycen_MP+1)
	#Account for rounding errors
	for uhh in range(len(xlist2)):
		if xlist2[uhh]>0 and xlist2[uhh]<0.1:
			xlist2[uhh]=0.
		if ylist2[uhh]>0 and ylist2[uhh]<0.1:
			ylist2[uhh]=0.
	xlist1=[(1.*xl2/(L_head['cdelt1']*3600.))+xcen_MP for xl2 in xlist2]
	ax.set_xticks(xlist1)
	if xon:
		ax.set_xticklabels(xlist2,rotation=0)
	else:
		ax.set_xticklabels([])
	ylist1=[(1.*yl2/(L_head['cdelt2']*3600.))+ycen_MP for yl2 in ylist2]
	ax.set_yticks(ylist1)
	if yon:
		ax.set_yticklabels(ylist2)
	else:
		ax.set_yticklabels([])
	overhang=0.1
	ax.set_xlim(((zooom+overhang)/(L_head['cdelt1']*3600.))+xcen_MP,((-1*zooom-overhang)/(L_head['cdelt1']*3600.))+xcen_MP)
	ax.set_ylim(((-1*zooom-overhang)/(L_head['cdelt2']*3600.))+ycen_MP,((zooom+overhang)/(L_head['cdelt2']*3600.))+ycen_MP)
	
	#Plot beam
	xb=0.15*(ax.get_xlim()[1]-ax.get_xlim()[0])+ax.get_xlim()[0]
	yb=ax.get_ylim()[0]+0.15*(ax.get_ylim()[1]-ax.get_ylim()[0])
	ell1=Ellipse(xy=[xb,yb],height=L_BMAJ_c*cfact,width=L_BMIN_c*cfact,angle=BPA,fill=True,facecolor='w',hatch='/')
	ax.add_artist(ell1)
	if wbar:
		#Plot rotation axes
		truecenter=[xcen_MP,ycen_MP]
		PA_r=(PA)*(np.pi/180.)
		ax_M_x=[truecenter[0]-(length/2.)*np.sin(PA_r),truecenter[0],truecenter[0]+(length/2.)*np.sin(PA_r)]
		ax_M_y=[truecenter[1]+(length/2.)*np.cos(PA_r),truecenter[1],truecenter[1]-(length/2.)*np.cos(PA_r)]
		ax.plot(ax_M_x,ax_M_y,color='k',linewidth=2)
		PA_m_r=(PA+90.)*(np.pi/180.)
		ax_m_x=[truecenter[0]-(length/2.)*np.sin(PA_m_r),truecenter[0],truecenter[0]+(length/2.)*np.sin(PA_m_r)]
		ax_m_y=[truecenter[1]+(length/2.)*np.cos(PA_m_r),truecenter[1],truecenter[1]-(length/2.)*np.cos(PA_m_r)]
		ax.plot(ax_m_x,ax_m_y,color='k',linestyle='dashed',linewidth=2)

#https://stackabuse.com/sorting-algorithms-in-python/#bubblesort
def bubble_sort(mat):
	bigarray=[]
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			if mat[i,j]!=0: 
				bigarray.append([mat[i,j],i,j])
	swapped = True
	while swapped:
		swapped = False
		for i in range(len(bigarray) - 1):
			if bigarray[i][0] > bigarray[i + 1][0]:
				bigarray[i],bigarray[i+1]=bigarray[i+1],bigarray[i]
				swapped = True
	return bigarray

#https://stackabuse.com/sorting-algorithms-in-python/#bubblesort
def bubble_sort_reverse(mat):
	bigarray=[]
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			if mat[i,j]!=0: 
				bigarray.append([mat[i,j],i,j])
	swapped = True
	while swapped:
		swapped = False
		for i in range(len(bigarray) - 1):
			if bigarray[i][0] < bigarray[i + 1][0]:
				bigarray[i],bigarray[i+1]=bigarray[i+1],bigarray[i]
				swapped = True
	return bigarray

#Takes in matrix of values (x) and current active space (y)
#Expands mask
#Outputs mask of >2sig pixels adjacent to AS
def getAS(matx,maty,sig):
	doneyet=False
	while not doneyet:
		changedsomething=False
		for i in range(1,matx.shape[0]-1):
			for j in range(1,matx.shape[1]-1):
				if maty[i,j]:
					for idiff in [-1,0,1]:
						for jdiff in [-1,0,1]:
							if idiff==0 and jdiff==0: 
								pass
							else:
								if (not maty[i+idiff,j+jdiff]) and matx[i+idiff,j+jdiff]>2*sig: 
									maty[i+idiff,j+jdiff]=1; changedsomething=True
		if changedsomething==False: 
			doneyet=True
	return maty

#Takes in cubemask of values (x) and current active space (y)
#Expands mask
#Outputs mask of >2sig pixels adjacent to AS
def getAS3D(matx,maty,sig):
	doneyet=False
	while not doneyet:
		changedsomething=False
		for i in range(1,matx.shape[0]-1):
			for j in range(1,matx.shape[1]-1):
				for k in range(1,matx.shape[2]-1):
					if maty[i,j,k]:
						for idiff in [-1,0,1]:
							for jdiff in [-1,0,1]:
								for kdiff in [-1,0,1]:
									if idiff==0 and jdiff==0 and kdiff==0: 
										pass
									else:
										if (not maty[i+idiff,j+jdiff,k+kdiff]) and matx[i+idiff,j+jdiff,k+kdiff]>2*sig: 
											maty[i+idiff,j+jdiff,k+kdiff]=1; changedsomething=True

		if changedsomething==False: 
			doneyet=True
	return maty

#Takes in FITS data and header, makes new FITS
def makenewcube(data_N,head_N,name_N):
	hdunew=fits.PrimaryHDU()
	hdunew.data=np.ones((data_N.shape[0],1,data_N.shape[1],data_N.shape[2]))
	hdunew.header['CDELT1']=head_N['CDELT1']
	hdunew.header['CDELT2']=head_N['CDELT2']
	hdunew.header['CDELT3']=1.
	hdunew.header['CDELT4']=head_N['CDELT3']
	hdunew.header['CRPIX1']=head_N['CRPIX1']
	hdunew.header['CRPIX2']=head_N['CRPIX2']
	hdunew.header['CRPIX3']=0.
	hdunew.header['CRPIX4']=head_N['CRPIX3']
	hdunew.header['CRVAL1']=head_N['CRVAL1']
	hdunew.header['CRVAL2']=head_N['CRVAL2']
	hdunew.header['CRVAL3']=0.
	hdunew.header['CRVAL4']=head_N['CRVAL3']
	hdunew.header['CTYPE1']=head_N['CTYPE1']
	hdunew.header['CTYPE2']=head_N['CTYPE2']
	hdunew.header['CTYPE3']='STOKES'
	hdunew.header['CTYPE4']=head_N['CTYPE3']
	hdunew.header['CUNIT1']=head_N['CUNIT1']
	hdunew.header['CUNIT2']=head_N['CUNIT2']
	hdunew.header['CUNIT3']=''
	hdunew.header['CUNIT4']=head_N['CUNIT3']
	#hdunew.header['FREQ0']=head_N['FREQ0']
	try:
		hdunew.header['RESTFRQ']=head_N['RESTFRQ']
	except NameError:
		print 'No RF'
	hdunew.header['SPECSYS']='LSRK'
	hdunew.header['BMAJ']=head_N['BMAJ']
	hdunew.header['BMIN']=head_N['BMIN']
	hdunew.header['BPA']=head_N['BPA']
	hdunew.header['BUNIT']=head_N['BUNIT']
	for misckey in ['EQUINOX','RADESYS','BSCALE','BZERO','BTYPE']:
		try:
			hdunew.header[misckey]=head_N[misckey]
		except KeyError:
			if misckey=='EQUINOX': hdunew.header[misckey]=2000.0
	hdunew.data[:,0,:,:]=data_N
	hdunew.writeto(name_N,overwrite=True)

#Takes in FITS data and header, makes new FITS, with no Stokes axis
def makenewcube_NS(data_N,head_N,name_N):
	hdunew=fits.PrimaryHDU()
	hdunew.data=np.ones((data_N.shape[0],data_N.shape[1],data_N.shape[2]))
	hdunew.header['CDELT1']=head_N['CDELT1']
	hdunew.header['CDELT2']=head_N['CDELT2']
	hdunew.header['CDELT3']=head_N['CDELT3']
	hdunew.header['CRPIX1']=head_N['CRPIX1']
	hdunew.header['CRPIX2']=head_N['CRPIX2']
	hdunew.header['CRPIX3']=head_N['CRPIX3']
	hdunew.header['CRVAL1']=head_N['CRVAL1']
	hdunew.header['CRVAL2']=head_N['CRVAL2']
	hdunew.header['CRVAL3']=head_N['CRVAL3']
	hdunew.header['CTYPE1']=head_N['CTYPE1']
	hdunew.header['CTYPE2']=head_N['CTYPE2']
	hdunew.header['CTYPE3']=head_N['CTYPE3']
	hdunew.header['CTYPE3']=head_N['CTYPE3']
	hdunew.header['CUNIT1']=head_N['CUNIT1']
	hdunew.header['CUNIT2']=head_N['CUNIT2']
	hdunew.header['CUNIT3']=head_N['CUNIT3']
	#hdunew.header['FREQ0']=head_N['FREQ0']
	try:
		hdunew.header['RESTFRQ']=head_N['RESTFRQ']
	except NameError:
		print 'No RF'
	hdunew.header['SPECSYS']='LSRK'
	hdunew.header['BMAJ']=head_N['BMAJ']
	hdunew.header['BMIN']=head_N['BMIN']
	hdunew.header['BPA']=head_N['BPA']
	hdunew.header['BUNIT']=head_N['BUNIT']
	for misckey in ['EQUINOX','RADESYS','BSCALE','BZERO','BTYPE']:
		try:
			hdunew.header[misckey]=head_N[misckey]
		except KeyError:
			if misckey=='EQUINOX': hdunew.header[misckey]=2000.0
	hdunew.data[:,:,:]=data_N
	hdunew.writeto(name_N,overwrite=True)

#Takes in FITS data and header, makes new FITS
def makenewcube2D(data_N,head_N,name_N):
	hdunew=fits.PrimaryHDU()
	hdunew.data=np.ones((data_N.shape[0],data_N.shape[1]))
	hdunew.header['CDELT1']=head_N['CDELT1']
	hdunew.header['CDELT2']=head_N['CDELT2']
	hdunew.header['CRPIX1']=head_N['CRPIX1']
	hdunew.header['CRPIX2']=head_N['CRPIX2']
	hdunew.header['CRVAL1']=head_N['CRVAL1']
	hdunew.header['CRVAL2']=head_N['CRVAL2']
	hdunew.header['CTYPE1']=head_N['CTYPE1']
	hdunew.header['CTYPE2']=head_N['CTYPE2']
	hdunew.header['CUNIT1']=head_N['CUNIT1']
	hdunew.header['CUNIT2']=head_N['CUNIT2']
	#hdunew.header['FREQ0']=head_N['FREQ0']
	hdunew.header['SPECSYS']='LSRK'
	hdunew.header['BMAJ']=head_N['BMAJ']
	hdunew.header['BMIN']=head_N['BMIN']
	hdunew.header['BPA']=head_N['BPA']
	hdunew.header['BUNIT']=head_N['BUNIT']
	for misckey in ['EQUINOX','RADESYS','BSCALE','BZERO','BTYPE']:
		try:
			hdunew.header[misckey]=head_N[misckey]
		except KeyError:
			if misckey=='EQUINOX': hdunew.header[misckey]=2000.0
	hdunew.data[:,:]=data_N
	hdunew.writeto(name_N,overwrite=True)

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta):
	xo = float(xo)
	yo = float(yo)	
	a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
	b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
	c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
	g = amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) + c*((y-yo)**2)))
	return g.ravel()

#Define general M_i
def Mi(f,x,xc,y,yc):
	temp=f*((x-xc)**2+(y-yc)**2)
	return temp

def nomore0(whichfits):
	#Import fits into CASA image
	ia=image()
	ia.fromfits('TEMP.im',whichfits,overwrite=True)
	#Mask all zeros
	ia.calcmask(mask='TEMP.im!=0.0',name='mymask1')
	#export as fits
	ia.tofits(whichfits,velocity=True,optical=False,dropdeg=True,dropstokes=True,overwrite=True)
	ia.done()

def chtoNU(m0c,L_head):
	temp=L_head['CRVAL3']+L_head['CDELT3']*(1+m0c-L_head['CRPIX3']) #m/s
	print '---->',VtoNU(temp,L_head['RESTFREQ'])
	return temp*(1E-3)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Data
	
#DATA IN HERE!
if 1==1:

	#No Line (1/15/2021)
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'CG12', 'sourcefile2':'CG12','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'CG14', 'sourcefile2':'CG14','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'CG21', 'sourcefile2':'CG21','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'CG38', 'sourcefile2':'CG38','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'CG42', 'sourcefile2':'CG42','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'CG47', 'sourcefile2':'CG47','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'CG75', 'sourcefile2':'CG75','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC274035', 'sourcefile2':'DC274035','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC351640', 'sourcefile2':'DC351640','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC357722', 'sourcefile2':'DC357722','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC378903', 'sourcefile2':'DC378903','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75}) #Nice L_Serendip (Aug2020)
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC400160', 'sourcefile2':'DC400160','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC403030', 'sourcefile2':'DC403030','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC416105', 'sourcefile2':'DC416105','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC430951', 'sourcefile2':'DC430951','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC510660', 'sourcefile2':'DC510660','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC536534', 'sourcefile2':'DC536534','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC628063', 'sourcefile2':'DC628063','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC665509', 'sourcefile2':'DC665509','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC665626', 'sourcefile2':'DC665626','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5}) #Nice L_Serendip (Aug2020)
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC680104', 'sourcefile2':'DC680104','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC709575', 'sourcefile2':'DC709575','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC722679', 'sourcefile2':'DC722679','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC742174', 'sourcefile2':'DC742174','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC803480', 'sourcefile2':'DC803480','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC814483', 'sourcefile2':'DC814483','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC834764', 'sourcefile2':'DC834764','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC842313_IMS', 'sourcefile2':'DC842313','PA':90, 'length':30, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.75}) #J1000 is in full cube (Jul2020)
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC843045', 'sourcefile2':'DC843045','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC845652', 'sourcefile2':'DC845652','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC859732', 'sourcefile2':'DC859732','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC0235', 'sourcefile2':'VC5101210235','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC4930', 'sourcefile2':'VC5101244930','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC8969', 'sourcefile2':'VC5101288969','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC5533', 'sourcefile2':'VC510605533','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})

	#Yes line, >1 ring (1/14/2021)
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'CG32_YF_RF', 'sourcefile2':'CG32','PA':242, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC396844_YF_RF', 'sourcefile2':'DC396844','PA':200, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC417567_YF_RF', 'sourcefile2':'DC417567','PA':50, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.8, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC432340_RF', 'sourcefile2':'DC432340','PA':180, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.1, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC434239_RF', 'sourcefile2':'DC434239','PA':218, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':True, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':False, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC454608_RF', 'sourcefile2':'DC454608','PA':190, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.7, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC494057_YF_RF', 'sourcefile2':'DC494057','PA':180, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[4.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC519281_RF', 'sourcefile2':'DC519281','PA':80, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.5,2.7], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.8, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC552206_YF_RF', 'sourcefile2':'DC552206','PA':315, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC627939_RF', 'sourcefile2':'DC627939','PA':285, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.2,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.8, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC683613_YF_RF', 'sourcefile2':'DC683613','PA':350, 'length':30, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.1,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC733857_RF', 'sourcefile2':'DC733857','PA':87, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.9, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True, 'delv2':200})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC773957_RF', 'sourcefile2':'DC773957','PA':35, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.55, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC818760_YF_RF', 'sourcefile2':'DC818760','PA':91, 'length':60, 'zooom':4.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC848185_GJ_RF', 'sourcefile2':'DC848185','PA':315, 'length':30, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[6.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC873321_RF', 'sourcefile2':'DC873321','PA':285, 'length':35, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.8, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC873756_YF_RF', 'sourcefile2':'DC873756','PA':300, 'length':30, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC881725_YF_RF', 'sourcefile2':'DC881725','PA':318, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC7582_RF', 'sourcefile2':'VC5100537582','PA':70, 'length':15, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC1407_RF', 'sourcefile2':'VC5100541407','PA':64, 'length':30, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.9, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC9223_RF', 'sourcefile2':'VC5100559223','PA':10, 'length':26, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True, 'delv2':200})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC2662_YF_RF', 'sourcefile2':'VC5100822662','PA':10, 'length':45, 'zooom':4.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.9, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC4794_YF_RF', 'sourcefile2':'VC5100994794','PA':235, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':300, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'delv2':200, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC9780_YF_RF', 'sourcefile2':'VC5101209780','PA':30, 'length':30, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC8326_YF_RF', 'sourcefile2':'VC5101218326','PA':10, 'length':30, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC6441_RF', 'sourcefile2':'VC510786441','PA':0, 'length':30, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.1, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True, 'delv2':200})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC7875_RF', 'sourcefile2':'VC5110377875','PA':143, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC6608_GJ_RF', 'sourcefile2':'VC5180966608','PA':310, 'length':30, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VE9038_YF_RF', 'sourcefile2':'VE530029038','PA':305, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	
	#Yes line, Unresolved (1/14/2021)	
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC308643', 'sourcefile2':'DC308643','PA':190, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC372292', 'sourcefile2':'DC372292','PA':0, 'length':25, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC422677_GJ', 'sourcefile2':'DC422677','PA':315, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC493583_YF', 'sourcefile2':'DC493583','PA':0, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC6653', 'sourcefile2':'VC510596653','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'VC9402_GJ', 'sourcefile2':'VC5100969402','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})

	#Yes line, 1 ring (1/14/21)	
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC488399_YF', 'sourcefile2':'DC488399','PA':269, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC494763', 'sourcefile2':'DC494763','PA':89, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC539609_YF', 'sourcefile2':'DC539609','PA':185, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC630594', 'sourcefile2':'DC630594','PA':120, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/ALPINE2020/', 'sourcefile':'DC880016', 'sourcefile2':'DC880016','PA':45, 'length':20, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
		
	#J1234 FIELD (Banerji et al. in prep)------------------------------------------------
	#N
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/J1234/', 'sourcefile':'NW1T_TRIM', 'sourcefile2':'J1234N_W1 (z=2.541)','PA':135, 'length':35, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[3.0,3.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co32', 'delv':1500, 'delv2':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'makepanels':False, 'mrradfact':1.0})
	#C
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/J1234/', 'sourcefile':'CW1T_TRIM', 'sourcefile2':'J1234C_W1 (z=2.503)','PA':335, 'length':35, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[3.0,3.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co32', 'delv':1500, 'delv2':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.7, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'makepanels':False, 'mrradfact':1.0})
	#S
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/J1234/', 'sourcefile':'SW1T_TRIM2', 'sourcefile2':'J1234S_W1 (z=2.497)','PA':10, 'length':35, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[3.0,3.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co32', 'delv':1500, 'delv2':250, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.7, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'makepanels':False, 'mrradfact':1.0})
	#J2315
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/Banerji/', 'sourcefile':'J2315_V1', 'sourcefile2':'J2315 (z=2.566)','PA':180, 'length':35, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,3.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co32', 'delv':750, 'delv2':250, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.0001, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'makepanels':False, 'mrradfact':0.8})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/Banerji/', 'sourcefile':'J2315_V1_N', 'sourcefile2':'J2315N','PA':225, 'length':20, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[3.0,3.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co32', 'delv':750, 'delv2':250, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.9, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'makepanels':False, 'mrradfact':0.7})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/Banerji/', 'sourcefile':'J2315_V1_S', 'sourcefile2':'J2315S','PA':180, 'length':13, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[3.0,3.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co32', 'delv':750, 'delv2':250, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.8, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'makepanels':False, 'mrradfact':0.8})
	
	#J17 GALAXIES-----------------------------------------------------------------------
	#Needs velocity averaging:
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/PURE/VAVG/', 'sourcefile':'Az159_FIXED_W2', 'sourcefile2':'AzTEC/C159 (z=4.57)','PA':174, 'length':30, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':250, 'PVR':True, 'vsysguessNum':1, 'VDFACT':2.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':False, 'ringfact':2.5})
	#bigdata.append(['/Volumes/SINNOH/Karim/AZ159/','AZ159_L_W1_S0_V1','AzTEC/C159 W1 (z=4.57)',170,30,999,999,1.,True,3.,2.5,False,999,999,'upper left','cii',500]);PVR=True;VDFACT=2.
	#bigdata.append(['/Volumes/SINNOH/Karim/AZ159/','AZ159_L_W2_S0_V1','AzTEC/C159 W2 (z=4.57)',170,30,999,999,1.,True,3.,2.5,False,999,999,'upper left','cii',500]);PVR=True;VDFACT=2.
	#bigdata.append(['/Volumes/SINNOH/Karim/AZ159/','AZ159_L_W4_S0_V1','AzTEC/C159 W4 (z=4.57)',170,30,999,999,1.,True,3.,2.5,False,999,999,'upper left','cii',500]);PVR=True;VDFACT=2.
	#bigdata.append(['/Volumes/SINNOH/Karim/AZ159/','AZ159_L_W8_S0_V1','AzTEC/C159 W8 (z=4.57)',170,30,999,999,1.,True,3.,2.5,False,999,999,'upper left','cii',500]);PVR=True;VDFACT=2.
	#bigdata.append(['/Volumes/SINNOH/Karim/AZ159/','AZ159_L_W16_S0_V1','AzTEC/C159 W16 (z=4.57)',170,30,999,999,1.,True,3.,2.5,False,999,999,'upper left','cii',500]);PVR=True;VDFACT=2.
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/PURE/VAVG/', 'sourcefile':'Az159_FIXED_W3', 'sourcefile2':'AzTEC/C159 (z=4.57)','PA':165, 'length':30, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.4], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':-1, 'PVR':True, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'fitvsys':False, 'ringfact':2.5, 'bbxlim2':True, 'mrradfact':1.0, 'unconinc':False, 'xy0':[-1,-1]})

	#Needs velocity averaging:
	#bigdata.append(['/Volumes/Kanto/FITS/PURE/','J1000_FIXED','COSMOS J100054+02343 (z=4.54)',146,40,50,55,1.0,True,3,2.5,False,999,999,'upper right','cii',500])
	#bigdata.append(['/Volumes/SINNOH/Karim/J1000/','J1000_S01_V1_L_W1','COSMOS J100054+02343 W1 (z=4.54)',146,40,999,999,1.0,True,3,2.5,False,999,999,'upper right','cii',500]);xy0=[133,143]
	#bigdata.append(['/Volumes/SINNOH/Karim/J1000/','J1000_S01_V1_L_W2','COSMOS J100054+02343 W2 (z=4.54)',146,40,999,999,1.0,True,3,2.5,False,999,999,'upper right','cii',500]);xy0=[133,143]
	#bigdata.append(['/Volumes/SINNOH/Karim/J1000/','J1000_S01_V1_L_W4','COSMOS J100054+02343 W4 (z=4.54)',146,40,999,999,1.0,True,3,2.5,False,999,999,'upper right','cii',500]);xy0=[133,143]
	#bigdata.append(['/Volumes/SINNOH/Karim/J1000/','J1000_S01_V1_L_W8','COSMOS J100054+02343 W8 (z=4.54)',146,40,999,999,1.0,True,3,2.5,False,999,999,'upper right','cii',500]);xy0=[133,143];fitvsys=False
	#bigdata.append(['/Volumes/SINNOH/Karim/J1000/','J1000_S0123_V1_L_W8','COSMOS J100054+02343 W8 (z=4.54)',146,40,999,999,1.0,True,3.5,2.7,False,999,999,'upper left','cii',700]);xy0=[133,143];ringfact=1.
	#bigdata.append(['/Volumes/SINNOH/Karim/J1000/','J1000_S01_V1_L_W16','COSMOS J100054+02343 W16 (z=4.54)',146,40,999,999,2.0,True,3,2.5,False,999,999,'upper right','cii',500]);xy0=[133,143]
	#bigdata.append(['/Volumes/SINNOH/Karim/J1000/','J1000_S01_V1_L_W32','COSMOS J100054+02343 W32 (z=4.54)',146,40,999,999,2.0,True,3,2.5,False,999,999,'upper right','cii',500]);xy0=[133,143]
	#bigdata.append({'sourcefold':'/Volumes/SINNOH/Karim/J1000/', 'sourcefile':'J1000_S0123_V1_L_W8', 'sourcefile2':'COSMOS J100054+02343 (z=4.54)','PA':140, 'length':25, 'zooom':1.3, 'HAVELINE':True, 'searchsnr':[5.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':-1, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'fitvsys':True, 'ringfact':2.5, 'bbxlim2':True, 'mrradfact':0.8, 'unconinc':False, 'xy0':[133,142]})
	#GOOD.
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/PURE/', 'sourcefile':'HZ9_FIXED', 'sourcefile2':'HZ9 (z=5.548)','PA':15, 'length':15, 'zooom':1.3, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5 })
	#GOOD.
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/PURE/', 'sourcefile':'HZ10_FIXED', 'sourcefile2':'HZ10 (z=5.659)','PA':290, 'length':30, 'zooom':1.5, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':1000, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim2':True })
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/PURE/', 'sourcefile':'J0817_FIXED2', 'sourcefile2':'J0817','PA':89, 'length':30, 'zooom':2.5, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':3.0, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'mrradfact':0.9})
	#GOOD.	
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/PURE/', 'sourcefile':'J1319_FIXED2', 'sourcefile2':'ULAS J131911.29+095051.4 (z=6.13)','PA':238, 'length':30, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim2':True})

	#MISC-----------------------------------------------------------------------
	#--COS3018
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/COS3018/haha/', 'sourcefile':'COS3018NAT_V', 'sourcefile2':'COS-3018555981 (z=6.85)','PA':60, 'length':40, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.5,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5})
	#--J0817
	#bigdata.append({'sourcefold':'/Volumes/Kanto/Neel/2017.1.01052.S/science_goal.uid___A001_X1273_X760/group.uid___A001_X1273_X761/member.uid___A001_X1273_X762/calibrated/', 'sourcefile':'MS12_UVCS0_LINE_W4_V0_RF', 'sourcefile2':'J0817','PA':109, 'length':60, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'unconinc':False})
	#--ALESS73.1
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/ALESS731/', 'sourcefile':'ALESS_TRIM_W8', 'sourcefile2':'ALESS 73.1 (z=4.756)','PA':45, 'length':55, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.5,2.3], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.5, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'unconinc':False})
	bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/ALESS731/', 'sourcefile':'ALESS_TRIM_W4', 'sourcefile2':'ALESS 73.1 (z=4.756)','PA':45, 'length':55, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.5,2.3], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.5, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'unconinc':False})

	#TANIO CUBES-----------------------------------------------------------------------	
	#bigdata.append({'sourcefold':'/Volumes/SINNOH/TanioOct2020/', 'sourcefile':'W0220_V1', 'sourcefile2':'W0220+0137 (z=3.1356)','PA':41, 'length':40, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':250, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.5, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'delv2':250, 'unconinc':False})
	#bigdata.append({'sourcefold':'/Volumes/SINNOH/TanioOct2020/', 'sourcefile':'W2246_V1', 'sourcefile2':'W2246-0526 (z=4.6009)','PA':100, 'length':40, 'zooom':1.3, 'HAVELINE':True, 'searchsnr':[3.0,2.4], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':True, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'unconinc':False})

	#Simon Dye data
	#AUG14
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/AUG14/', 'sourcefile':'h2o', 'sourcefile2':'ID141 H2O_211202 AUG 14 [L]','PA':110, 'length':80, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[51,44], 'fitvsys':False, 'ringfact':2.5, 'DATA_MASK':'MASK'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/AUG14/', 'sourcefile':'h2o', 'sourcefile2':'ID141 H2O_211202 AUG 14 [A]','PA':110, 'length':80, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'azim', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[51,44], 'fitvsys':False, 'ringfact':2.5, 'DATA_MASK':'MASK'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/AUG14/', 'sourcefile':'co76', 'sourcefile2':'ID141 CO(7-6) AUG 14 [A]','PA':110, 'length':80, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.5], 'normtype':'azim', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':2.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[51,44], 'fitvsys':False, 'ringfact':2.75, 'resm0':[-1,-1,-1], 'm0box':"26,21,68,58"})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/AUG14/', 'sourcefile':'co76', 'sourcefile2':'ID141 CO(7-6) AUG 14 [L]','PA':110, 'length':80, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':2.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':False, 'ringfact':2.75})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/AUG14/', 'sourcefile':'h2o_sub1', 'sourcefile2':'ID141 H2O_211202 AUG 14 [A]','PA':110, 'length':25, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.5], 'normtype':'azim', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':False, 'ringfact':2.75})
	#AUG31
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/AUG31/', 'sourcefile':'main', 'sourcefile2':'ID141 Main','PA':110, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[50,41], 'fitvsys':False, 'ringfact':1.0, 'resm0':[0.373,0.171,104.5]})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/AUG31/', 'sourcefile':'secondary', 'sourcefile2':'ID141 Secondary','PA':110, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[42,64], 'fitvsys':True, 'ringfact':1.0, 'resm0':[0.440,0.186,102.3]})
	#Oct12
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/Oct12/', 'sourcefile':'sie2sm_main_b4_spw0_recon-uv_source_cube', 'sourcefile2':'ID141 Main (Oct 12)','PA':110, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'azim', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[51,46], 'fitvsys':False, 'ringfact':2.5, 'resm0':[-1,-1,-1], 'DATA_MASK':'MASK'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/Oct12/', 'sourcefile':'sie2sm_secondary_b4_spw0_recon-uv_source_cube', 'sourcefile2':'ID141 Secondary (Oct12)','PA':110, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper left', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':4, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[33,60], 'fitvsys':True, 'ringfact':1.0, 'resm0':[0.3,0.2,100], 'DATA_MASK':'MASK'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/Oct12/', 'sourcefile':'main_RedMain2', 'sourcefile2':'ID141 Secondary (Oct 12, -Main model)','PA':110, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[36,62], 'fitvsys':True, 'ringfact':2.5, 'resm0':[0.3,0.1,116], 'DATA_MASK':'MASK'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/Oct12/', 'sourcefile':'main_RedMain2_E', 'sourcefile2':'ID141 Secondary, E (Oct 12, -Main model)','PA':160, 'length':30, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':250, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'resm0':[-1,-1,-1], 'DATA_MASK':'MASK'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/Oct12/', 'sourcefile':'main_RedMain2_W', 'sourcefile2':'ID141 Secondary, W (Oct 12, -Main model)','PA':50, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.0,2.0], 'normtype':'local', 'LR':'upper left', 'whichline':'co76', 'delv':250, 'PVR':True, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[45,65], 'fitvsys':True, 'ringfact':2.5, 'resm0':[-1,-1,-1], 'DATA_MASK':'MASK'})
	#Oct20
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT20/', 'sourcefile':'temp3', 'sourcefile2':'ID141 H2O_211202 OCT 20 [L]','PA':110, 'length':80, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[51,44], 'fitvsys':False, 'ringfact':2.5, 'DATA_MASK':'MASK'})
	
	#12/7/20
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'co_full_v2', 'sourcefile2':'ID141 CO (Local)','PA':100, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':4.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[51,43], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.8, 'MASKTYPE':'SEARCH', 'bbxlim2':True})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'h2o_full_v1', 'sourcefile2':'ID141 H2O (Local)','PA':100, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':3.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[51,43], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.8, 'MASKTYPE':'SEARCH', 'bbxlim2':True})
	#SPLIT
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'co_main_v2', 'sourcefile2':'ID141 CO (Azimuthal, Main)','PA':100, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'azim', 'LR':'upper left', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':7.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[51,43], 'fitvsys':False, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.8, 'MASKTYPE':'SEARCH', 'bbxlim2':True})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'h2o_main_v1', 'sourcefile2':'ID141 H2O (Azimuthal, Main)','PA':100, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'azim', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':5.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[51,43], 'fitvsys':False, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.8, 'MASKTYPE':'SEARCH', 'bbxlim2':True})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'CO_IMMATH_DEC07', 'sourcefile2':'ID141 CO (Local, Secondary)','PA':120, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.7], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':7.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[34,60], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.8, 'MASKTYPE':'SEARCH'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'H2O_IMMATH_DEC07', 'sourcefile2':'ID141 H2O (Local, Secondary)','PA':115, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.3], 'normtype':'local', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':5.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[34,60], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.8, 'MASKTYPE':'SEARCH'})
	
	#Oct23
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'co_main_v1', 'sourcefile2':'ID141 CO_MAIN OCT23 [L]','PA':112, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[5.6,3.2], 'normtype':'local', 'LR':'upper center', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':2.5, 'fitz':4.24417, 'upsidedown':True, 'xy0':[50,41], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(1E-4)})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'co_secondary_v1', 'sourcefile2':'ID141 CO_SECONDARY OCT23 [A]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[8.8,5.0], 'normtype':'azim', 'LR':'upper left', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[43,64], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(1E-4)})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'h2o_main_v0', 'sourcefile2':'ID141 H2O_MAIN OCT23 [L]','PA':110, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[5.7,3.3], 'normtype':'local', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.3, 'fitz':4.24417, 'upsidedown':False, 'xy0':[51,41], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(5E-5)})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'h2o_secondary_v0', 'sourcefile2':'ID141 H2O_SECONDARY OCT23 [A]','PA':110, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[8.0,4.6], 'normtype':'azim', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.1, 'fitz':4.24417, 'upsidedown':False, 'xy0':[34,60], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(5E-5)})

	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'CO_MAIN_R0_L', 'sourcefile2':'ID141 CO_SECONDARY OCT23 [L, -main]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[7.2,4.1], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[43,64], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(1E-4)})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'CO_MAIN_R0_A', 'sourcefile2':'ID141 CO_SECONDARY OCT23 [A, -main]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[7.2,4.1], 'normtype':'azim', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[43,64], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(1E-4)})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'H2O_MAIN_R0_L', 'sourcefile2':'ID141 H2O_SECONDARY OCT23 [L, -main]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[6.6,3.7], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[34,60], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(5E-5)})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'H2O_MAIN_R0_A', 'sourcefile2':'ID141 H2O_SECONDARY OCT23 [A, -main]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[6.7,3.8], 'normtype':'azim', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[34,60], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(5E-5)})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'CO_MAIN_R0_L_E', 'sourcefile2':'ID141 CO_SECONDARY OCT23 [L, -main, E]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[6.7,3.8], 'normtype':'azim', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(1E-4)})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'CO_MAIN_R0_L_W', 'sourcefile2':'ID141 CO_SECONDARY OCT23 [L, -main, W]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[6.7,3.8], 'normtype':'azim', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(1E-4)})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'H2O_MAIN_R0_L_E', 'sourcefile2':'ID141 H2O_SECONDARY OCT23 [L, -main, E]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[6.7,3.8], 'normtype':'azim', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(1E-4)})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/OCT23/', 'sourcefile':'H2O_MAIN_R0_L_W', 'sourcefile2':'ID141 H2O_SECONDARY OCT23 [L, -main, W]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[6.7,3.8], 'normtype':'azim', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'THRESHOLD', 'THRESHOLD':2.5*(1E-4)})

	#Nov06
	#AZIM
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'co_main_v2', 'sourcefile2':'ID141 CO_MAIN NOV06 [A]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'azim', 'LR':'upper center', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':3.5, 'fitz':4.24417, 'upsidedown':True, 'xy0':[50,41], 'fitvsys':False, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'co_secondary_v1', 'sourcefile2':'ID141 CO_SECONDARY NOV06 [A]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'azim', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':3.5, 'fitz':4.24417, 'upsidedown':True, 'xy0':[37,63], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'h2o_main_v1', 'sourcefile2':'ID141 H2O_MAIN NOV06 [A]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'azim', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.1, 'fitz':4.24417, 'upsidedown':False, 'xy0':[52,42], 'fitvsys':False, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'h2o_secondary_v1', 'sourcefile2':'ID141 H2O_SECONDARY NOV06 [A]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'azim', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':5.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[34,60], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	#LOCAL
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'co_main_v2', 'sourcefile2':'ID141 CO_MAIN NOV06 [L]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper center', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':10., 'fitz':4.24417, 'upsidedown':True, 'xy0':[51,42], 'fitvsys':False, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'co_secondary_v1', 'sourcefile2':'ID141 CO_SECONDARY NOV06 [L]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper left', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':3.5, 'fitz':4.24417, 'upsidedown':True, 'xy0':[37,63], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'h2o_main_v1', 'sourcefile2':'ID141 H2O_MAIN NOV06 [L]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':2.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[52,42], 'fitvsys':False, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'h2o_secondary_v1', 'sourcefile2':'ID141 H2O_SECONDARY NOV06 [L]','PA':110, 'length':60, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':5.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[34,60], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.8, 'MASKTYPE':'SEARCH'})

	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'CO_FINALSUB2_L', 'sourcefile2':'ID141 CO_SECONDARY NOV06 [L, MINUS]','PA':110, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[37,63], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'CO_FINALSUB2_A', 'sourcefile2':'ID141 CO_SECONDARY NOV06 [A, MINUS]','PA':130, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':True, 'xy0':[37,63], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'H2O_FINALSUB2_L', 'sourcefile2':'ID141 H2O_SECONDARY NOV06 [L, MINUS]','PA':100, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'H2O_FINALSUB2_A', 'sourcefile2':'ID141 H2O_SECONDARY NOV06 [A, MINUS]','PA':100, 'length':50, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'h20211202', 'delv':500, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.1, 'fitz':4.24417, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH'})
	
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'H2O_FINALSUB2_A_E', 'sourcefile2':'ID141 H2O_2nd_E NOV06 [L]','PA':130, 'length':40, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.0,2.25], 'normtype':'local', 'LR':'upper right', 'whichline':'h20211202', 'delv':250, 'PVR':False, 'vsysguessNum':3, 'VDFACT':0.8, 'fitz':4.24417, 'upsidedown':False, 'xy0':[20.2,56.1], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH', 'delv2':150})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'H2O_FINALSUB2_A_W', 'sourcefile2':'ID141 H2O_2nd_W NOV06 [L]','PA':10, 'length':40, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.0,2.25], 'normtype':'local', 'LR':'upper left', 'whichline':'h20211202', 'delv':250, 'PVR':True, 'vsysguessNum':3, 'VDFACT':0.9, 'fitz':4.24417, 'upsidedown':False, 'xy0':[45.4,63.6], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH', 'delv2':150})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'CO_FINALSUB2_A_E', 'sourcefile2':'ID141 CO_2nd_E NOV06 [L]','PA':100, 'length':40, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.0,2.25], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':250, 'PVR':False, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[20.2,56.1], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH', 'delv2':150})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ID141/NOV06/', 'sourcefile':'CO_FINALSUB2_A_W', 'sourcefile2':'ID141 CO_2nd_W NOV06 [L]','PA':100, 'length':40, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.0,2.25], 'normtype':'local', 'LR':'upper right', 'whichline':'co76', 'delv':250, 'PVR':False, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[45.4,63.6], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH', 'delv2':150})

	#Feb 25, 2021 - [CI] Test
	#bigdata.append({'sourcefold':'/Volumes/Sinnoh/ID141/', 'sourcefile':'ci_V3', 'sourcefile2':'ID141 CI (PRELIM) [L]','PA':110, 'length':40, 'zooom':0.3, 'HAVELINE':True, 'searchsnr':[3.0,1.5], 'normtype':'local', 'LR':'upper right', 'whichline':'ci', 'delv':250, 'PVR':False, 'vsysguessNum':3, 'VDFACT':1.0, 'fitz':4.24417, 'upsidedown':False, 'xy0':[52,46], 'fitvsys':True, 'ringfact':2.5, 'DATA_MASK':'MASK', 'mrradfact':0.7, 'MASKTYPE':'SEARCH', 'delv2':150})

	#Aravena Test
	#TOO WEAK
	#bigdata.append(['/Volumes/Kanto/AravenaALMA/2018.1.01359.S/science_goal.uid___A001_X133d_X2fe2/group.uid___A001_X133d_X2fe3/member.uid___A001_X133d_X2fe4/product/','HZ7_V0','HZ7 (z=5.250)',260,40,999,999,1.0,True,3,2.5,False,999,999,'upper right','cii',500])
	#TOO WEAK
	#bigdata.append(['/Volumes/Kanto/AravenaALMA/2018.1.01359.S/science_goal.uid___A001_X133d_X2fee/group.uid___A001_X133d_X2fef/member.uid___A001_X133d_X2ff0/product/','CLM1_V0','CLM1 (z=???)',260,40,999,999,1.0,True,3,2.5,False,999,999,'upper right','cii',500])
	#bigdata.append(['/Volumes/Kanto/AravenaALMA/2018.1.01359.S/science_goal.uid___A001_X133d_X2fe6/group.uid___A001_X133d_X2fe7/member.uid___A001_X133d_X2fe8/product/','HZ9_V0','HZ9',20,40,999,999,1.0,True,3,2.5,False,999,999,'upper right','cii',500]);fitvsys=False

	#Aravena Real
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ARAV/', 'sourcefile':'COS2987_COMB_CUBE_30_IMSUB', 'sourcefile2':'COS2987 (z=6.81)','PA':170, 'length':40, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.2,3.2], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'delv2':100, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[147,136], 'fitvsys':True, 'ringfact':2.0, 'makepanels':False, 'mrradfact':0.3, 'linw':1})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ARAV/', 'sourcefile':'CLM1_COMB_CUBE_10_IM', 'sourcefile2':'CLM1 W1 (z=6.17)','PA':220, 'length':40, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'delv2':100, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'makepanels':False, 'mrradfact':0.5, 'linw':1})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ARAV/', 'sourcefile':'CLM1_COMB_CUBE_10_IMS_W2', 'sourcefile2':'CLM1 W2 (z=6.17)','PA':220, 'length':40, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'delv2':100, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'makepanels':False, 'mrradfact':0.5, 'linw':1})	
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ARAV/', 'sourcefile':'CLM1_COMB_CUBE_10_IMS_W3', 'sourcefile2':'CLM1 W3 (z=6.17)','PA':220, 'length':40, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'delv2':100, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'makepanels':False, 'mrradfact':0.5, 'linw':1})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ARAV/', 'sourcefile':'COS2987_combined_CUBE_IMSUB', 'sourcefile2':'COS2987 COMBINED (z=6.81)','PA':170, 'length':40, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.1,3.1], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'delv2':100, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'makepanels':False, 'mrradfact':0.5, 'linw':1})

	#8/18/2020 Test
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/supercode/', 'sourcefile':'ALESS_ASYMM', 'sourcefile2':'ASYMM TEST','PA':45, 'length':55, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.5,2.0], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':False, 'ringfact':2.75 })

	#March 29, 2021 - Edo Ibar Tests
	#bigdata.append({'sourcefold':'/Users/garethjones/Downloads/Delivered_products_Ibar_ALPINE_20210215/DC818760/', 'sourcefile':'DC818760_Ibar_1', 'sourcefile2':'DC818760 Ibar 1','PA':91, 'length':90, 'zooom':4.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.6, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Users/garethjones/Downloads/Delivered_products_Ibar_ALPINE_20210215/DC818760/', 'sourcefile':'DC818760_Ibar_2', 'sourcefile2':'DC818760 Ibar 2','PA':91, 'length':90, 'zooom':4.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Users/garethjones/Downloads/Delivered_products_Ibar_ALPINE_20210215/DC873756/', 'sourcefile':'DC873756_Ibar_1', 'sourcefile2':'DC873756 Ibar 1','PA':210, 'length':80, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.2, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Users/garethjones/Downloads/Delivered_products_Ibar_ALPINE_20210215/DC873756/', 'sourcefile':'DC873756_Ibar_2', 'sourcefile2':'DC873756 Ibar 2','PA':300, 'length':90, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper right', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Users/garethjones/Downloads/Delivered_products_Ibar_ALPINE_20210215/VC8326/', 'sourcefile':'VC8326_Ibar_1', 'sourcefile2':'VC8326 Ibar 1','PA':10, 'length':90, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':0.6, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Users/garethjones/Downloads/Delivered_products_Ibar_ALPINE_20210215/VC8326/', 'sourcefile':'VC8326_Ibar_2', 'sourcefile2':'VC8326 Ibar 2','PA':10, 'length':90, 'zooom':3.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})

	#April 1, 2021 - Tanio additional cubes
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/W0134_W0831/', 'sourcefile':'W0134', 'sourcefile2':'W0134','PA':95, 'length':70, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':2.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/W0134_W0831/', 'sourcefile':'W0831', 'sourcefile2':'W0831','PA':175, 'length':60, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':True, 'vsysguessNum':3, 'VDFACT':2.0, 'fitz':-1, 'upsidedown':False, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':2.5, 'bbxlim':True})

	#Cy B12
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/ALMA_CY8/Cy8_B12/', 'sourcefile':'B12N_SO1_out', 'sourcefile2':'B12N SO1 (Round 2)','PA':260, 'length':60, 'zooom':1.0, 'HAVELINE':True, 'searchsnr':[3.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'PVR':True, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'xy0':[-1,-1], 'fitvsys':True, 'ringfact':1.0, 'bbxlim':True})

	#Michele Hot DOG (8/25/21)
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/michele/', 'sourcefile':'B4_L_V0_IMS2', 'sourcefile2':'W0410-0913 (z=3.63)','PA':132, 'length':70, 'zooom':1.3, 'HAVELINE':True, 'searchsnr':[5.0,2.0], 'normtype':'local', 'LR':'upper left', 'whichline':'co65', 'delv':500, 'delv2':500, 'PVR':False, 'vsysguessNum':1, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'xy0':[48,50], 'fitvsys':True, 'ringfact':2.5, 'bbxlim2':True, 'mrradfact':2.0})

	#B13 [CII] (9/1/21)
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/FITS/B13CII/', 'sourcefile':'B13_LINE_V0_CLEAN_W4_IMS', 'sourcefile2':'BRI1335-0417 [CII] (W4)','PA':5, 'length':70, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[5.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':-1, 'PVR':False, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'fitvsys':True, 'ringfact':1.0, 'bbxlim2':True, 'mrradfact':1.0, 'unconinc':False})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/FITS/B13CII/', 'sourcefile':'B13_LINE_V0_CLEAN_W10_IMS', 'sourcefile2':'BRI1335-0417 (z=4.407)','PA':16, 'length':70, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[5.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':-1, 'PVR':False, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'fitvsys':True, 'ringfact':1.0, 'bbxlim2':True, 'mrradfact':1.0, 'unconinc':False})
	#bigdata.append({'sourcefold':'/Users/garethjones/Desktop/FITS/B13CII/', 'sourcefile':'B13_LINE_V0_CLEAN_W16_IMS', 'sourcefile2':'BRI1335-0417 (z=4.407)','PA':5, 'length':70, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[5.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':-1, 'PVR':False, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'fitvsys':True, 'ringfact':2.5, 'bbxlim2':True, 'mrradfact':1.0, 'unconinc':False})
	#bigdata.append({'sourcefold':'/Volumes/Izalith/B13CII/2017.1.00394.S/science_goal.uid___A001_X1284_X1b20/group.uid___A001_X1284_X1b21/member.uid___A001_X1284_X1b22/calibrated/', 'sourcefile':'B13_LINE_V0_CLEAN_W24', 'sourcefile2':'BRI1335-0417 [CII] (W24)','PA':5, 'length':70, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[5.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':-1, 'PVR':False, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'fitvsys':True, 'ringfact':2.5, 'bbxlim2':True, 'mrradfact':1.0, 'unconinc':False})

	#Jan tests (10/22/21)
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/JanCubes/', 'sourcefile':'w0149_jan', 'sourcefile2':'BRI1335-0417 [CII] (W24)','PA':5, 'length':70, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[5.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':-1, 'PVR':False, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'fitvsys':True, 'ringfact':2.5, 'bbxlim2':True, 'mrradfact':1.0, 'unconinc':False})	
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/JanCubes/', 'sourcefile':'w0410_jan', 'sourcefile2':'BRI1335-0417 [CII] (W24)','PA':5, 'length':70, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[5.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':-1, 'PVR':False, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'fitvsys':True, 'ringfact':2.5, 'bbxlim2':True, 'mrradfact':1.0, 'unconinc':False})	
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/JanCubes/', 'sourcefile':'w2238_jan', 'sourcefile2':'BRI1335-0417 [CII] (W24)','PA':5, 'length':70, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[5.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':-1, 'PVR':False, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'fitvsys':True, 'ringfact':2.5, 'bbxlim2':True, 'mrradfact':1.0, 'unconinc':False})	
	#bigdata.append({'sourcefold':'/Volumes/Kanto/FITS/JanCubes/', 'sourcefile':'w2305_jan', 'sourcefile2':'BRI1335-0417 [CII] (W24)','PA':5, 'length':70, 'zooom':2.0, 'HAVELINE':True, 'searchsnr':[5.0,2.5], 'normtype':'local', 'LR':'upper left', 'whichline':'cii', 'delv':500, 'delv2':-1, 'PVR':False, 'vsysguessNum':2, 'VDFACT':1.0, 'fitz':-1, 'upsidedown':True, 'fitvsys':True, 'ringfact':2.0, 'bbxlim2':True, 'mrradfact':0.8, 'unconinc':False})	


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~	
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#Real code

#RUN PROPERTIES!!!!!!!!!!!!!!!!!
dobb=True
multiVD=False
MODELFACTOR=0.01  #1E-2

for i in range(len(bigdata)):

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#Import data
	bigdata_snip=bigdata[i]
	sourcefold=bigdata_snip['sourcefold']
	sourcefile=bigdata_snip['sourcefile']
	sourcefile2=bigdata_snip['sourcefile2']
	PA=bigdata_snip['PA']
	length=bigdata_snip['length']
	zooom=bigdata_snip['zooom']
	HAVELINE=bigdata_snip['HAVELINE']
	searchsnr=[bigdata_snip['searchsnr'][0],bigdata_snip['searchsnr'][1]]
	LR=bigdata_snip['LR']
	whichline=bigdata_snip['whichline']
	delv=bigdata_snip['delv']
	PVR=bigdata_snip['PVR']
	vsysguessNum=bigdata_snip['vsysguessNum']
	VDFACT=bigdata_snip['VDFACT']
	fitz=bigdata_snip['fitz']
	upsidedown=bigdata_snip['upsidedown']
	fitvsys=bigdata_snip['fitvsys']
	ringfact=bigdata_snip['ringfact']
	normtype=bigdata_snip['normtype']
	#
	try:
		xy0=bigdata_snip['xy0']
	except KeyError:
		print 'Using center of image.'
	#
	try:
		resm0=bigdata_snip['resm0']
	except KeyError:
		print 'Making no assumptions on M0 fit'
	#
	try:
		delv2=bigdata_snip['delv2']
	except KeyError:
		print 'Using default Delv2'
	#
	try:
		makepanels=bigdata_snip['makepanels']
		print 'Making each panel separately'
	except KeyError:
		print 'Making one big plot'
	#
	try:
		mrradfact=bigdata_snip['mrradfact']
		print 'Using maxringrad factor of',mrradfact
	except KeyError:
		print 'Using maxringrad factor of',mrradfact,'(default)'
	#
	try:
		linw=bigdata_snip['lw']
		print 'Using linewidth',linw
	except KeyError:
		print 'Using linewidth',linw
	#
	try:
		m0box=bigdata_snip['m0box']
		print 'Custom m0box'
	except KeyError:
		print 'Using full m0box'
	#
	try:
		DATA_MASK=bigdata_snip['DATA_MASK']
		print 'Comparing model to',DATA_MASK
	except KeyError:
		print 'Comparing model to',DATA_MASK,'(Default)'
	#
	try:
		MASKTYPE=bigdata_snip['MASKTYPE']
		print 'Using a ',MASKTYPE,' mask'
	except KeyError:
		print 'Using a SEARCH mask (Default)'
	#
	try:
		THRESHOLD=bigdata_snip['THRESHOLD']
		print 'Using threshold of',THRESHOLD
	except KeyError:
		print 'No Threshold info (Default)'
	#
	try:
		bbxlim=bigdata_snip['bbxlim']
		print 'Using spectral xlim of -750 to 750'
	except KeyError:
		pass
	#
	try:
		bbxlim2=bigdata_snip['bbxlim2']
		print 'Using spectral xlim of -1000 to 1000'
	except KeyError:
		pass
	if (not bbxlim) and (not bbxlim2):
		print 'Using full spectral range'
	#
	try:
		unconinc=bigdata_snip['unconinc']
		print 'Using educated inc guess'
	except KeyError:
		print 'Using inc range of 10-80, guess 45 (Default)'


	if whichline=='cii':
		linefreq=1900.5369E+9
	elif whichline=='ci21':
		linefreq=809.34197000E+9
	elif whichline=='co76':
		linefreq=806.65180600E+9
	elif whichline=='h20211202':
		linefreq=752.03314300E+9
	elif whichline=='co65':
		linefreq=691.4730763E+9
	elif whichline=='co32':
		linefreq=345.79598990E+9
	elif whichline=='co10':
		linefreq=115.27120180E+9
	else:
		print 'NO REST FREQ'

	#Set PVD separation (zoomnum)
	if zooom==0.3:
		xlist2=[-0.2,0.0,0.2]
		ylist2=[-0.2,0.0,0.2]
		zoomnum=0.1
	if zooom==0.7:
		xlist2=[-0.5,0.0,0.5]
		ylist2=[-0.5,0.0,0.5]
		zoomnum=0.2
	if zooom==1.0:
		xlist2=[-0.5,0.0,0.5]
		ylist2=[-0.5,0.0,0.5]
		zoomnum=0.5
	if zooom==1.5 or zooom==1.4 or zooom==1.3:
		xlist2=[-1.0,0.0,1.0]
		ylist2=[-1.0,0.0,1.0]
		zoomnum=0.8
	if zooom==2.0:
		xlist2=[-1.0,0.0,1.0]
		ylist2=[-1.0,0.0,1.0]
		zoomnum=0.8
	if zooom==2.5:
		xlist2=[-2.0,0.0,2.0]
		ylist2=[-2.0,0.0,2.0]
		zoomnum=1.0
	if zooom==3.0:
		xlist2=[-2.0,0.0,2.0]
		ylist2=[-2.0,0.0,2.0]
		zoomnum=1.0
	if zooom==4.0:
		xlist2=[-3.0,0.0,3.0]
		ylist2=[-3.0,0.0,3.0]
		zoomnum=2.0
			
	print 'Working on',sourcefile2

	#Clean up.
	badfiles=[outputfolder+sourcefile+'_L_SIG.fits',outputfolder+sourcefile+'_L_NOI.fits',outputfolder+sourcefile+'_L_SIG_RED.fits',outputfolder+sourcefile+'_L_SIG_RED2.fits',outputfolder+sourcefile+'_L_FUL.fits',outputfolder+sourcefile+'_L_MASK.fits',outputfolder+sourcefile+'_MODEL.fits',outputfolder+sourcefile+'_MODEL_m0_script.fits',outputfolder+sourcefile+'_MODEL_m1_script.fits',outputfolder+sourcefile+'_MODEL_m2_script.fits',outputfolder+sourcefile+'NONEmod_'+normtype+'.fits',outputfolder+sourcefile+'NONEmod_'+normtype+'2.fits',outputfolder+sourcefile+'NONE_mom0th.fits',outputfolder+sourcefile+'NONE_mom1st.fits',outputfolder+sourcefile+'NONE_mom2nd.fits',outputfolder+'pyscript.py',outputfolder+'gnuscript.gnu',outputfolder+'densprof.txt',outputfolder+RINGFILE+'1.txt',outputfolder+RINGFILE+'2.txt',outputfolder+'mask.fits',outputfolder+'detections.txt',outputfolder+'DetectMap.fits',outputfolder+'detections.fits',sourcefold+sourcefile+'bigplot.png',sourcefold+sourcefile2+'_spectrum.txt',sourcefold+sourcefile+'_PV1_M_script.fits',sourcefold+sourcefile+'_PV2_M_script.fits',sourcefold+sourcefile+'_PV1_script.fits',sourcefold+sourcefile+'_PV2_script.fits',sourcefold+sourcefile+'_m2_script_SIG_RED.fits',sourcefold+sourcefile+'_m0_script_FUL_NOI.fits',sourcefold+sourcefile+'_m0_script_FUL_SIG_RED.fits',sourcefold+sourcefile+'_m1_script_SIG_RED.fits',sourcefold+sourcefile+'_m0_script_FUL.fits',sourcefold+'detections.fits',sourcefold+'detections.txt',sourcefold+'DetectMap.fits',sourcefold+'NONE_mom0th.fits',sourcefold+'NONE_mom1st.fits',sourcefold+'NONE_mom2nd.fits']
	badfolds=[outputfolder+sourcefile+'_MODEL.im',outputfolder+sourcefile+'_MODELm0.im',outputfolder+sourcefile+'_MODELm1.im',outputfolder+sourcefile+'_MODELm2.im',outputfolder+'maps',outputfolder+'pvs',sourcefold+sourcefile+'_m2_script_SIG_RED.im',sourcefold+sourcefile+'_m0_script_FUL_NOI.im',sourcefold+sourcefile+'_m0_script_FUL_SIG_RED.im',sourcefold+sourcefile+'_m1_script_SIG_RED.im',sourcefold+sourcefile+'_m0_script_FUL.im']
	for badfile in badfiles:
		try:
			os.remove(badfile)
			print "Deleted: ",badfile
		except OSError:
			pass
	for badfolder in badfolds:
		try:
			shutil.rmtree(badfolder)
			print "Deleted: ",badfolder
		except OSError:
			pass

	#Get input cube name
	cubename=sourcefold+sourcefile+'.fits'

	#Get rid of zeros
	nomore0(cubename)

	#Get basic output fits files
	mom0file_name=sourcefold+sourcefile+'_m0_script.fits'
	mom1file_name=sourcefold+sourcefile+'_m1_script.fits'
	mom2file_name=sourcefold+sourcefile+'_m2_script.fits'
	PV1file_name=sourcefold+sourcefile+'_PV1_script.fits'
	PV2file_name=sourcefold+sourcefile+'_PV2_script.fits'

	#Import line fits into matrix
	L_head=fits.getheader(cubename,0)
	if L_head['CUNIT3']=='Hz':
		try:
			cubename2=cubename.replace('.fits','_V.fits')
			hdu_L = fits.open(cubename2)
			L_head=fits.getheader(cubename2,0)
			cubename=cubename.replace('.fits','_V.fits')
		except IOError:
			ia=image()
			ia.fromfits(cubename.replace('.fits','_V.im'),cubename,overwrite=True)
			ia.tofits(cubename.replace('.fits','_V.fits'),velocity=True,optical=False,dropdeg=True,dropstokes=True)
			cubename=cubename.replace('.fits','_V.fits')
	hdu_L = fits.open(cubename)
	L_head=fits.getheader(cubename,0)
	fixcube3d(cubename,999,cubename)
	L_data=hdu_L[0].data
	#Get rid of stokes, correct order of axes, make nans into zeros
	if len(L_data.shape)==4:
		ZLEN=L_data.shape[1];YLEN=L_data.shape[2];XLEN=L_data.shape[3]
		temp_L=np.zeros((XLEN,YLEN,ZLEN))
		for i in range(XLEN):
			for j in range(YLEN):
				for k in range(ZLEN):
					if fn.isGood(L_data[0,k,j,i]): 
						temp_L[i,j,k]=L_data[0,k,j,i]
	if len(L_data.shape)==3:
		ZLEN=L_data.shape[0];YLEN=L_data.shape[1];XLEN=L_data.shape[2]
		temp_L=np.zeros((XLEN,YLEN,ZLEN))
		for i in range(XLEN):
			for j in range(YLEN):
				for k in range(ZLEN):
					if fn.isGood(L_data[k,j,i]): 
						temp_L[i,j,k]=L_data[k,j,i]

	#If central position isn't given, assume it's at the image center
	if xy0[0]<0:
		xy1=[XLEN/2.,YLEN/2.]
	else:
		xy1=[xy0[0],xy0[1]]
 
	#Get some values
	try:
		L_BMAJ_c=L_head['BMAJ']/abs(L_head['CDELT1'])/cfact #px
		BMAJ=L_head['BMAJ']
		L_BMIN_c=L_head['BMIN']/abs(L_head['CDELT1'])/cfact #px
		BMIN=L_head['BMIN']
		BPA=L_head['BPA']
	except KeyError: #Catch multi-beam images
		print 'MB'
		L_BMAJ_c=avgbeam(hdu_L[1].data)[0]/abs(L_head['CDELT1'])/cfact #px
		BMAJ=avgbeam(hdu_L[1].data)[0]
		L_BMIN_c=avgbeam(hdu_L[1].data)[1]/abs(L_head['CDELT1'])/cfact #px
		BMIN=avgbeam(hdu_L[1].data)[1]
		BPA=avgbeam(hdu_L[1].data)[2]
	w=WCS(L_head)
	#Clean up
	hdu_L.close()

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#Search line cube, make set of .fits, make moment images

	if MASKTYPE=='THRESHOLD':
		M0emission=False
		SIGNAL_MASK=np.zeros((XLEN,YLEN,ZLEN)) #x y z
		putitback_N=np.zeros((ZLEN,YLEN,XLEN)) #To make noise (data-signal) cube
		putitback_S=np.zeros((ZLEN,YLEN,XLEN)) #To make reduced signal cube & Stokes-free cube
		putitback_MASK=np.zeros((ZLEN,YLEN,XLEN)) #To make reduced signal MASK cube
		m0_ch_low,m0_ch_high=[1E+10,-1E+10]
		for i in range(XLEN):
			for j in range(YLEN):
				for k in range(ZLEN):
					if temp_L[i,j,k]>=THRESHOLD:
						M0emission=True
						SIGNAL_MASK[i,j,k]=1
						putitback_S[k,j,i]=L_data[k,j,i]
					else:
						putitback_N[k,j,i]=L_data[k,j,i]
					if temp_L[i,j,k]>=THRESHOLD*(3./2.5):
							if k>m0_ch_high: 
								m0_ch_high=k
							if k<m0_ch_low: 
								m0_ch_low=k


	elif MASKTYPE=='SEARCH':
		#Run BBSearch
		parname='/Users/garethjones/Documents/'+sourcefile+'_search_script.par'
		f=open(parname,'w'); f.close()
		f=open(parname,'w')
		f.write('FITSFILE\t'+cubename+'\n')
		f.write('OUTFOLDER\t'+sourcefold+'\n')
		f.write('checkChannels\tTrue\n')
		f.write('minchannels\t2\n')
		f.write('SEARCH\t\tTrue\n')
		f.write('STATS\t\tTrue\n')
		f.write('SNRCUT\t\t'+str(searchsnr[0])+'\n')
		f.write('GROWTHCUT\t\t'+str(searchsnr[1])+'\n')
		f.close()
		os.system('cd '+BBFOLD+';'+BBCOMMAND+' -p '+parname)
		#Get detections.fits mask
		if os.path.isfile(sourcefold+'detections.fits') :
			detectionsfits=sourcefold+'detections.fits'
		else:
			with gzip.open(sourcefold+'detections.fits.gz', 'rb') as f_in:
				with open(sourcefold+'detections.fits', 'wb') as f_out:
					shutil.copyfileobj(f_in, f_out)
			detectionsfits=sourcefold+'detections.fits'
		hdu_SIGNAL = fits.open(sourcefold+'detections.fits')
		SIGNAL_data=hdu_SIGNAL[0].data #z x y
		SIGNAL_header=fits.getheader(sourcefold+'detections.fits',0)
		hdu_SIGNAL.close()
		#Do rough M0
		roughm0=np.zeros((XLEN,YLEN))
		for i in range(XLEN):
			for j in range(YLEN):
				for k in range(ZLEN):
					if fn.isGood(SIGNAL_data[k,j,i]):
						roughm0[i,j]+=SIGNAL_data[k,j,i]
		#Check if central area (radius of 25px) is nonzero
		anyline=False
		for xr in range(XLEN):
			for yr in range(YLEN):
				xdist=xr-xy1[0]; ydist=yr-xy1[1]; xydist=np.sqrt(xdist**2+ydist**2)
				if xydist<25.:
					if roughm0[xr,yr]!=0.0:
						M0emission=True
						anyline=True
		if not anyline:
			M0emission=False; mom0success=False
			G_M0=None;M20_M0=None
		#SIGNAL_MASK=np.zeros((XLEN,YLEN,ZLEN)) #x y z
		putitback_N=np.zeros((ZLEN,YLEN,XLEN)) #To make noise (data-signal) cube
		putitback_S=np.zeros((ZLEN,YLEN,XLEN)) #To make reduced signal cube & Stokes-free cube
		putitback_MASK=np.zeros((ZLEN,YLEN,XLEN)) #To make reduced signal MASK cube
		m0_ch_low,m0_ch_high=[1E+10,-1E+10]
		if M0emission:
			#Get full reduced detection mask
			Dmat=np.zeros((ZLEN,YLEN,XLEN)) #Original SIGNAL_data
			Mmat=np.zeros((ZLEN,YLEN,XLEN)) #Output reduced data
			for i in range(XLEN):
				for j in range(YLEN):
					for k in range(ZLEN):
						Dmat[k,j,i]=SIGNAL_data[k,j,i]
			for xr in range(XLEN):
				for yr in range(YLEN):
					xdist=xr-xy1[0]; ydist=yr-xy1[1]; xydist=np.sqrt(xdist**2+ydist**2)
					if xydist<XYDIST:
						for zr in range(ZLEN):
							if Dmat[zr,yr,xr]!=0:
								Tmat=np.zeros((ZLEN,YLEN,XLEN))
								Tmat[zr,yr,xr]=1
								Tmat2=getAS3D(SIGNAL_data,Tmat,0)
								for xri in range(XLEN):
									for yrj in range(YLEN):
										for zrk in range(ZLEN): 
											if Tmat2[zrk,yrj,xri]>0:
												Dmat[zrk,yrj,xri]=0
												Mmat[zrk,yrj,xri]=1
			#Apply this mask to data cube to get reduced signal cube & get total noise cube
			for i in range(XLEN):
				for j in range(YLEN):
					for k in range(ZLEN):
						if Mmat[k,j,i]>0:
							putitback_S[k,j,i]=L_data[k,j,i]
							putitback_MASK[k,j,i]=1
							if k>m0_ch_high: 
								m0_ch_high=k
							if k<m0_ch_low: 
								m0_ch_low=k
						if SIGNAL_data[k,j,i]==0 and fn.isGood(L_data[k,j,i]):
							putitback_N[k,j,i]=L_data[k,j,i]

		#Get nonreduced signal cube
		completemask=np.zeros((ZLEN,YLEN,XLEN))
		for i in range(XLEN):
			for j in range(YLEN):
				for k in range(ZLEN):
					if SIGNAL_data[k,j,i]!=0:
						completemask[k,j,i]=SIGNAL_data[k,j,i]

	vsysguess=float(np.average([fn.CHtoV(L_head,m0_ch_low),fn.CHtoV(L_head,m0_ch_high)]))
	RF=L_head['RESTFRQ']
	
	#Write out full, masked, central masked, and noise .fits
	#
	print 'cd '+sourcefold+';cp '+cubename+' '+outputfolder+sourcefile+'_L_FUL.fits'
	os.system('cd '+sourcefold+';cp '+cubename+' '+outputfolder+sourcefile+'_L_FUL.fits') #Complete L cube
	fixcube3d(outputfolder+sourcefile+'_L_FUL.fits',RF,outputfolder+sourcefile+'_L_FUL.fits')
	#
	if MASKTYPE=='SEARCH':
		makenewcube(completemask,L_head,sourcefold+'L_SIG.fits')
		if os.path.exists(outputfolder+sourcefile+'_L_SIG.fits'):
			os.system('rm '+outputfolder+sourcefile+'_L_SIG.fits')
		print 'cd '+sourcefold+';mv L_SIG.fits '+outputfolder+sourcefile+'_L_SIG.fits'	
		os.system('cd '+sourcefold+';mv L_SIG.fits '+outputfolder+sourcefile+'_L_SIG.fits') #Masked L cube (3/2)
	#
	makenewcube_NS(putitback_MASK,L_head,sourcefold+'L_MASK.fits')
	if os.path.exists(outputfolder+sourcefile+'_L_MASK.fits'):
		os.system('rm '+outputfolder+sourcefile+'_L_MASK.fits')
	print 'cd '+sourcefold+';mv L_MASK.fits '+outputfolder+sourcefile+'_L_MASK.fits'
	os.system('cd '+sourcefold+';mv L_MASK.fits '+outputfolder+sourcefile+'_L_MASK.fits') #Mask for L cube (3/2), with only central source
	#
	makenewcube(putitback_S,L_head,sourcefold+'L_SIG_RED.fits')
	if os.path.exists(outputfolder+sourcefile+'_L_SIG_RED.fits'):
		os.system('rm '+outputfolder+sourcefile+'_L_SIG_RED.fits')
	print 'cd '+sourcefold+';mv L_SIG_RED.fits '+outputfolder+sourcefile+'_L_SIG_RED.fits'
	os.system('cd '+sourcefold+';mv L_SIG_RED.fits '+outputfolder+sourcefile+'_L_SIG_RED.fits') #Masked L cube (3/2), with only central source
	#	
	makenewcube_NS(putitback_S,L_head,sourcefold+'L_SIG_RED2.fits')
	if os.path.exists(outputfolder+sourcefile+'_L_SIG_RED2.fits'):
		os.system('rm '+outputfolder+sourcefile+'_L_SIG_RED2.fits')
	print 'cd '+sourcefold+';mv L_SIG_RED2.fits '+outputfolder+sourcefile+'_L_SIG_RED2.fits'
	os.system('cd '+sourcefold+';mv L_SIG_RED2.fits '+outputfolder+sourcefile+'_L_SIG_RED2.fits') #Masked L cube (3/2), with only central source and no stokes
	#
	makenewcube(putitback_N,L_head,sourcefold+'L_NOI.fits')
	if os.path.exists(outputfolder+sourcefile+'_L_NOI.fits'):
		os.system('rm '+outputfolder+sourcefile+'_L_NOI.fits')
	print 'cd '+sourcefold+';mv L_NOI.fits '+outputfolder+sourcefile+'_L_NOI.fits'
	os.system('cd '+sourcefold+';mv L_NOI.fits '+outputfolder+sourcefile+'_L_NOI.fits') #Noise of masked L cube (3/2)

	#Get cube rms/ch
	rmslist=[]
	for k in range(ZLEN):	
		temp=fn.getrms_basic(putitback_N[k,:,:])
		if temp>0: 
			rmslist.append(temp)
	chrms=np.mean(rmslist)

	if M0emission:

		#Make moment 0 map from whole cube, using channels with signal
		m0name_im=mom0file_name.replace('.fits','_FUL.im')
		ia=image()
		ia.open(outputfolder+sourcefile+'_L_FUL.fits')
		subim=ia.subimage(dropdeg=True,overwrite=True)
		rg=regionmanager()
		r=rg.box(blc=[0,0,m0_ch_low],trc=[XLEN-1,YLEN-1,m0_ch_high])
		m0im=subim.moments(moments=[0],outfile=m0name_im,overwrite=True,region=r)		
		mom0file_name=m0name_im.replace('.im','.fits')
		m0im.tofits(outfile=mom0file_name,overwrite=True)
		m0im.done()
		subim.done()
		ia.done()
		'''
		fixcube(mom0file_name,999,mom0file_name)
		ia=image()
		ia.fromfits(m0name_im.replace('_FUL.im','_FUL2.im'),mom0file_name,overwrite=True)
		ia.done()
		m0name_im=m0name_im.replace('_FUL.im','_FUL2.im')
		'''
		hdu_M = fits.open(mom0file_name)
		M_data=hdu_M[0].data; M_head=fits.getheader(mom0file_name,0)
		hdu_M.close()
		temp_M=np.zeros((XLEN,YLEN))
		for i in range(XLEN):
			for j in range(YLEN):
				temp_M[i,j]=M_data[j,i]
		#Using signal from BBSearch, make moment 0 map
		m0name_im_S=mom0file_name.replace('.fits','_SIG_RED.im')
		ia=image()
		ia.open(outputfolder+sourcefile+'_L_SIG_RED.fits')
		subim=ia.subimage(dropdeg=True,overwrite=True)
		rg=regionmanager()
		m0im=subim.moments(moments=[0],outfile=m0name_im_S,overwrite=True,region=r)
		mom0file_name_S=m0name_im_S.replace('.im','.fits')
		m0im.tofits(outfile=mom0file_name_S,overwrite=True)
		m0im.done()
		subim.done()
		ia.done()
		fixcube(mom0file_name_S,999,mom0file_name_S)
		hdu_M = fits.open(mom0file_name_S)
		M_data=hdu_M[0].data
		hdu_M.close()
		temp_M_S=np.zeros((XLEN,YLEN))
		for i in range(XLEN):
			for j in range(YLEN):
				temp_M_S[i,j]=M_data[j,i]
		mom0_rms=fn.getrms_basic(temp_M_S)

		#Using noise from BBSearch, make noise moment 0 map (only used for RMS)
		m0name_im_N=mom0file_name.replace('.fits','_NOI.im')
		ia=image()
		ia.open(outputfolder+sourcefile+'_L_NOI.fits')
		subim=ia.subimage(dropdeg=True,overwrite=True)
		rg=regionmanager()
		m0im=subim.moments(moments=[0],outfile=m0name_im_N,overwrite=True,region=r)
		mom0file_name_N=m0name_im_N.replace('.im','.fits')	
		m0im.tofits(outfile=mom0file_name_N,overwrite=True)
		m0im.done()
		subim.done()
		ia.done()
		fixcube(mom0file_name_N,999,mom0file_name_N)
		hdu_M = fits.open(mom0file_name_N)
		M_data=hdu_M[0].data
		hdu_M.close()
		temp_M_N=np.zeros((XLEN,YLEN))
		for i in range(XLEN):
			for j in range(YLEN):
				temp_M_N[i,j]=M_data[j,i]
		mom0_rms=fn.getrms_basic(temp_M_N)

		sigmask=np.zeros((XLEN,YLEN))
		for i in range(XLEN):
			for j in range(YLEN):
				if temp_M[i,j]>0.0*mom0_rms:
					sigmask[i,j]=1 

		#Using channels from previous step, make moment 1 map
		m1name_im=mom1file_name.replace('.fits','_SIG_RED.im')
		ia=image()
		ia.open(outputfolder+sourcefile+'_L_SIG_RED.fits')
		subim=ia.subimage(dropdeg=True,overwrite=True)
		r=rg.box(blc=[0,0,m0_ch_low],trc=[XLEN-1,YLEN-1,m0_ch_high])
		m1im=subim.moments(moments=[1],outfile=m1name_im,region=r,overwrite=True)
		mom1file_name=m1name_im.replace('.im','.fits')
		m1im.tofits(outfile=mom1file_name,overwrite=True)
		m1im.done()
		subim.done()
		ia.done()
		hdu_M1 = fits.open(mom1file_name)
		M1_data=hdu_M1[0].data; M1_head=fits.getheader(mom1file_name,0)
		hdu_M1.close()
		XLEN1=M1_data.shape[1]; YLEN1=M1_data.shape[0]
		temp_M1=np.zeros((XLEN1,YLEN1))
		vsysguess2=M1_data[xy1[1],xy1[0]]
		for i in range(XLEN1):
			for j in range(YLEN1):
				if 1==1:
					temp_M1[i,j]=M1_data[j,i]-float(vsysguess2)		

		#Using channels from previous step, make moment 2 map
		m2name_im=mom2file_name.replace('.fits','_SIG_RED.im')
		ia=image()
		ia.open(outputfolder+sourcefile+'_L_SIG_RED.fits')
		subim=ia.subimage(dropdeg=True,overwrite=True)
		r=rg.box(blc=[0,0,m0_ch_low],trc=[XLEN-1,YLEN-1,m0_ch_high])
		m2im=subim.moments(moments=[2],outfile=m2name_im,region=r,overwrite=True)
		mom2file_name=m2name_im.replace('.im','.fits')
		m2im.tofits(outfile=mom2file_name,overwrite=True)
		m2im.done()
		subim.done()
		ia.done()
		hdu_M2 = fits.open(mom2file_name)
		M2_data=hdu_M2[0].data; M2_head=fits.getheader(mom2file_name,0)
		hdu_M2.close()
		temp_M2=np.zeros((XLEN,YLEN))
		for i in range(XLEN):
			for j in range(YLEN):
				if 1==1:
					temp_M2[i,j]=M2_data[j,i]

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#ANALYSIS

		#Attempt 2D Fit
		M_est=open('tempest.txt','w'); M_est.close()
		M_est=open('tempest.txt','w')	
		if resm0[0]==-1:
			M_est.write(str(fn.matminmax(temp_M_S)[1])+', '+str(xy1[0])+', '+str(xy1[1])+', '+str(M_head['BMAJ']*3600)+'arcsec, '+str(M_head['BMIN']*3600.)+'arcsec, '+str(M_head['BPA'])+'deg')
		else:
			M_est.write(str(fn.matminmax(temp_M_S)[1])+', '+str(xy1[0])+', '+str(xy1[1])+', '+str(resm0[0])+'arcsec, '+str(resm0[1])+'arcsec, '+str(resm0[2])+'deg')
		M_est.close()
		ia=image()
		ia.open(m0name_im)
		print 'Fitting',m0name_im
		if m0box=='-1':
			temp=ia.fitcomponents(estimates='tempest.txt',logfile='/Users/garethjones/Desktop/supercode/FITLOG.txt') #,includepix=[-1E+10,1E+10]
		else:
			temp=ia.fitcomponents(estimates='tempest.txt',box=m0box,logfile='/Users/garethjones/Desktop/supercode/FITLOG.txt') #,includepix=[-1E+10,1E+10]
		try:
			beampx=temp['deconvolved']['component0']['beam']['beampixels']
		except KeyError:
			try:
				beampx=temp['results']['component0']['beam']['beampixels']
			except KeyError:
				beampx=999
		ia.done()
		mom0success=temp['converged'] and temp['results']['component0']['peak']['value']>3*temp['results']['component0']['peak']['error']
		#Debug:
		#print temp
		#
		print "Mom0 fit successful? "+str(mom0success)#+'('+temp['converged']+temp['results']['component0']['peak']['value']>3*temp['results']['component0']['peak']['error']+')'
		
		#Catch casses where mom0success is a length-1 array
		try:
			deleter=len(mom0success)
			mom0success=mom0success[0]
		except TypeError:
			pass

		print 'mom0success',mom0success,': Is it True?',mom0success==True


		if mom0success:
			#Integrated flux [Jy km/s]
			M_intflux=temp['results']['component0']['flux']['value'][0]
			M_intflux_e=temp['results']['component0']['flux']['error'][0]
			#Peak flux density [Jy/bm km/s]
			M_pkflux=temp['results']['component0']['peak']['value']
			M_pkflux_e=temp['results']['component0']['peak']['error']
			
			#Catch case where best-fit source is pt
			if not temp['deconvolved']['component0']['ispoint']: 
				#Deconvolved model FWHM ['']
				M_FWHM_maj=temp['deconvolved']['component0']['shape']['majoraxis']['value']
				M_FWHM_maj_e=temp['deconvolved']['component0']['shape']['majoraxiserror']['value']
				M_FWHM_min=temp['deconvolved']['component0']['shape']['minoraxis']['value']
				M_FWHM_min_e=temp['deconvolved']['component0']['shape']['minoraxiserror']['value']
				#Deconvolved model angle [deg]
				M_PA=temp['deconvolved']['component0']['shape']['positionangle']['value']
				M_PA_e=temp['deconvolved']['component0']['shape']['positionangleerror']['value']
			else:
				#Deconvolved model FWHM ['']
				M_FWHM_maj=1E+10
				M_FWHM_maj_e=1E+10
				M_FWHM_min=1E+10
				M_FWHM_min_e=1E+10
				#Deconvolved model angle [deg]
				M_PA=1E+10
				M_PA_e=1E+10

			#Convolved model FWHM ['']
			M_FWHM_maj_C=temp['results']['component0']['shape']['majoraxis']['value']
			M_FWHM_maj_e_C=temp['results']['component0']['shape']['majoraxiserror']['value']
			M_FWHM_min_C=temp['results']['component0']['shape']['minoraxis']['value']
			M_FWHM_min_e_C=temp['results']['component0']['shape']['minoraxiserror']['value']
			#Convolved model angle [deg]
			M_PA_C=temp['results']['component0']['shape']['positionangle']['value']
			M_PA_e_C=temp['results']['component0']['shape']['positionangleerror']['value']
			#Convolved model central RA [rad] & uncertainty ['']
			M_RA_rad=temp['results']['component0']['shape']['direction']['m0']['value']
			M_RA_as_e=temp['results']['component0']['shape']['direction']['error']['longitude']['value']
			#Convolved model central DEC [rad] & uncertainty ['']
			M_DEC_rad=temp['results']['component0']['shape']['direction']['m1']['value']
			M_DEC_as_e=temp['results']['component0']['shape']['direction']['error']['latitude']['value']

			M_Ampl=unp.uarray(M_pkflux,M_pkflux_e) #mJy/bm
			tempvel=w.wcs_pix2world([[0,0,ZLEN/2]],0)[0][2]
			M_RA_px=w.wcs_world2pix([[M_RA_rad/(np.pi/180.),M_DEC_rad/(np.pi/180.),tempvel]],0)[0][0]
			M_DEC_px=w.wcs_world2pix([[M_RA_rad/(np.pi/180.),M_DEC_rad/(np.pi/180.),tempvel]],0)[0][1]
			M_RA_px_e=M_RA_as_e/(3600.*np.abs(M_head['CDELT1']))
			M_DEC_px_e=M_DEC_as_e/(3600.*np.abs(M_head['CDELT2']))
			M_x0=unp.uarray(M_RA_px,M_RA_px_e) #px
			M_y0=unp.uarray(M_DEC_px,M_DEC_px_e) #px
			M_fwhmx=unp.uarray(M_FWHM_maj,M_FWHM_maj_e) #arcsec		
			M_fwhmy=unp.uarray(M_FWHM_min,M_FWHM_min_e) #arcsec
			M_theta=unp.uarray(M_PA,M_PA_e) #deg
			M_int=unp.uarray(M_intflux,M_intflux_e) #mJy
			inclguess=np.arccos(M_FWHM_min/M_FWHM_maj)*180/np.pi

			if xy0[0]<0:
				xy0[0]=M_RA_px
				xy0[1]=M_DEC_px

			if VDFACT==-1:
				VD_GUESS=-1
			else:
				VD_GUESS=temp_M2[xy0[0],xy0[1]]/VDFACT

		else:
			print 'M0success failed:',mom0success
			dobb=False

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#Do G-M20 of line	
		if (not M0emission):
			G_M0=None
			M20_M0=None
			print 'No Emission'
		else:
			print 'Getting G-M20...'
			#Get G of moment 0
			tempval=bubble_sort(temp_M_S)
			M0LIST_INC=[]
			for i in range(len(tempval)): M0LIST_INC.append(tempval[i][0])
			G_M0=0
			for i in range(len(M0LIST_INC)): 
				i2=i+1
				G_M0+=M0LIST_INC[i]*(2*i2-len(M0LIST_INC)-1)
			G_M0*=(1/(np.mean(M0LIST_INC)*len(M0LIST_INC)*(len(M0LIST_INC)-1)))
			#
			#Get M20 of moment 0
			M0LIST_DEC=bubble_sort_reverse(temp_M_S)
			#Find xc,yc that minimze Mi 
			smallestxcyc=[1E+10,0,0]
			for ic in range(temp_M_S.shape[0]):
				for jc in range(temp_M_S.shape[1]):
					Mtot=0
					for i in range(len(M0LIST_DEC)):
						Mtot+=Mi(M0LIST_DEC[i][0],M0LIST_DEC[i][1],ic,M0LIST_DEC[i][2],jc)
					if smallestxcyc[0]>Mtot:
						smallestxcyc=[Mtot,ic,jc] #[Mtot, xc, yc]
			#Start summing Mi until sum(fi)=0.2Ftot
			ftot=0
			for i in range(len(M0LIST_DEC)): ftot+=M0LIST_DEC[i][0]
			current_ftot=0
			current_Mi=0
			i=0
			while True:
				current_Mi+=Mi(M0LIST_DEC[i][0],M0LIST_DEC[i][1],smallestxcyc[1],M0LIST_DEC[i][2],smallestxcyc[2])
				current_ftot+=M0LIST_DEC[i][0]
				i+=1
				if current_ftot>0.2*ftot:
					break
			M20_M0=np.log10(current_Mi/smallestxcyc[0])

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		if dobb: #Do BB fit
			print 'Performing BB Fit'
			VDISP=VD_GUESS
			#Get parameter file
			parname=BBFOLD+sourcefile+'_script.par'
			f=open(parname,'w');f.close();f=open(parname,'w')
			#1f.write('FITSFILE\t'+sourcefold+sourcefile+'.fits'+'\n')
			f.write('FITSFILE\t'+outputfolder+sourcefile+'_L_FUL.fits'+'\n')
			f.write('PLOTS\tfalse\n')
			f.write('OUTFOLDER\t'+outputfolder+'\n')
			f.write('3DFIT\t\tTrue\n')
			minringrad=3600.*BMIN/ringfact
			try:
				if M_FWHM_maj<800:
					maxringrad=M_FWHM_maj*mrradfact
				else:
					print 'BAD MAXRINGRAD_1'
					maxringrad=0.
			except NameError:
				maxringrad=0.
				print 'BAD MAXRINGRAD_2'
			if maxringrad>minringrad:
				nradii=int(round(maxringrad/minringrad))
				radsep=round(maxringrad/float(nradii),2)
			else:
				print 'TOOOOOOO SMALLLLLLLLL',maxringrad,minringrad
				nradii=0
				radsep=999
			#numrad=max(2,int(round(float(maxringrad)/float(minringrad),0)))
			print minringrad,maxringrad,maxringrad/minringrad,'->',radsep,nradii
			f.write('NRADII\t\t'+str(nradii)+'\n')
			f.write('checkChannels\tTrue\n')
			f.write('SHOWBAR\t\tTrue\n')
			f.write('VERBOSE\t\tTrue\n')
			f.write('RADSEP\t\t'+str(radsep)+'\n')
			if vsysguessNum==1:
				f.write('VSYS\t\t0\n')
			elif vsysguessNum==2:
				f.write('VSYS\t\t'+str(vsysguess)+'\n')
			elif vsysguessNum==3:
				f.write('VSYS\t\t'+str(vsysguess2)+'\n')
			elif vsysguessNum==4:
				f.write('VSYS\t\t'+str(100.)+'\n')
			if xy1[0]==XLEN/2. and xy1[1]==YLEN/2.:
				try:
					f.write('XPOS\t\t'+str(round(M_RA_px,2))+'\n')
					f.write('YPOS\t\t'+str(round(M_DEC_px,2))+'\n')
				except NameError:
					f.write('XPOS\t\t'+str(round(xy0[0],2))+'\n')
					f.write('YPOS\t\t'+str(round(xy0[1],2))+'\n')
			else:
					f.write('XPOS\t\t'+str(round(xy0[0],2))+'\n')
					f.write('YPOS\t\t'+str(round(xy0[1],2))+'\n')

			f.write('VROT\t\t'+str(vrotguess)+'\n')
			f.write('VDISP\t\t'+str(VDISP)+'\n')
			if unconinc:
				f.write('INC\t\t\t45\n'); f.write('DELTAINC\t35\n')
			else:
				f.write('INC\t\t\t'+str(np.arccos(M_FWHM_min/M_FWHM_maj)*180/np.pi)+'\n'); f.write('DELTAINC\t20\n')
			f.write('PA\t\t\t'+str(PA)+'\n')
			#f.write('PA\t\t\t-1\n')
			f.write('Z0\t\t\t0.01\n')
			f.write('SMOOTH\t\tFalse\n')
			f.write('FLAGERRORS\tTrue\n')
			if nradii==1 or (not dotwostage): 
				f.write('TWOSTAGE\tFalse\n')
			else: 
				f.write('TWOSTAGE\tTrue\n')
			f.write('REGTYPE\t\tbezier\n') #auto, bezier, median, 0/1/2/...
			f.write('NORM\t\t'+normtype+'\n')
			f.write('SIDE\t\tB\n') #B -AR
			f.write('FTYPE\t\t2\n') #2
			f.write('WFUNC\t\t1\n') #2
			f.write('LTYPE\t\t1\n') #1
			f.write('SIDE\t\t'+SIDE+'\n') #A/R/B
			#f.write('CUMULATIVE\t\tTrue\n')
			if fitvsys:
				f.write('FREE\t\tVROT VDISP INC PA VSYS\n') 
			else:
				f.write('FREE\t\tVROT VDISP INC PA\n') 

			#f.write('MASK\t\tfile('+outputfolder+sourcefile+'_L_MASK.fits)\n')
			if MASKTYPE=='SEARCH':
				f.write('MASK\t\tSEARCH\n')
				f.write('SEARCH\t\tTrue\n')
				f.write('SNRCUT\t\t'+str(searchsnr[0])+'\n')
				f.write('GROWTHCUT\t'+str(searchsnr[1])+'\n')	
			if MASKTYPE=='THRESHOLD':
				f.write('MASK\t\tTHRESHOLD\n')
				f.write('THRESHOLD\t\t'+str(THRESHOLD)+'\n')
			f.write('MINCHANNELS\t\t2\n')				
			#f.write('MASK\t\tNONE\n')
			#f.write('flagRobustStats\tFalse\n')	
			f.write('SPACEPAR\tFalse\n')
			f.close()
			#
			os.system('cd '+BBFOLD+';pwd;'+BBCOMMAND+' -p '+sourcefile+'_script.par')
			#
			if normtype=='none':normtype='nonorm'

			#Make output cube readable
			fixcube3d(outputfolder+'NONEmod_'+normtype+'.fits',RF,outputfolder+'NONEmod_'+normtype+'2.fits')
			try:
				os.remove(outputfolder+sourcefile+'_MODEL.fits')
			except OSError:
				pass
			os.system('cd '+outputfolder+'; cp NONEmod_'+normtype+'2.fits '+outputfolder+sourcefile+'_MODEL.fits') 
			
			#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			#Decode BB outputs

			if fitsig: sigi=2
			else: sigi=0

			if fitvsys: sigvsys=2
			else: sigvsys=0

			RUN1=np.genfromtxt(outputfolder+RINGFILE+'1.txt',skip_header=1)

			INavg,INstd=getweighted(4,13+sigvsys+sigi)
			PAavg,PAstd=getweighted(5,15+sigvsys+sigi)

			if (not nradii==1) and dotwostage:
				print 'GETTING',float(nradii),'RING RESULTS'

				#BB gets the VS in a mystical method, so until we figure it out...
				nothing,VSstd=getweighted(11,17+sigvsys+sigi)
				VSavg=np.genfromtxt(outputfolder+RINGFILE+'2.txt',skip_header=1)[0][11]
				RUN2=np.genfromtxt(outputfolder+RINGFILE+'2.txt',skip_header=1)
				VRLIST_E1=[abs(x) for x in RUN2[:,13]]
				VRLIST_E2=RUN2[:,14]
				VDS_E1=abs(RUN2[:,15])
				VDS_E2=abs(RUN2[:,16])
				VDS=RUN2[:,3]
				XLIST=np.zeros(len(VRLIST_E1))
				VLIST=np.zeros(len(VRLIST_E1))
				VRLIST=np.zeros(len(VRLIST_E1))
				VDS_ERR=np.zeros(len(VRLIST_E1))
				for i in range(len(VRLIST_E1)): 
					XLIST[i]=str(RUN2[i,1]) #arcsec
					VLIST[i]=str(RUN2[i,2]) #km/s
					VRLIST[i]=max(VRLIST_E1[i],VRLIST_E2[i]) #km/s
					VDS_ERR[i]=max(VDS_E1[i],VDS_E2[i]) #km/s
			elif nradii==1:
				print 'GETTING 1 RING RESULTS...'
				#BB gets the VS in a mystical method, so until we figure it out...
				VSavg=abs(RUN1[11])
				VRLIST_E1=abs(RUN1[13])
				VRLIST_E2=RUN1[14]
				VDS_E1=abs(RUN1[15])
				VDS_E2=abs(RUN1[16])
				VDS=RUN1[3]
				XLIST=str(RUN1[1]) #arcsec
				VLIST=str(RUN1[2]) #km/s
				VRLIST=max(VRLIST_E1,VRLIST_E2) #km/s
				VDS_ERR=max(VDS_E1,VDS_E2) #km/s
			else:
				print 'GETTING ONE STAGE RESULTS...'
				#BB gets the VS in a mystical method, so until we figure it out...
				VSavg,VSstd=getweighted(11,17+sigvsys+sigi)
				VRLIST_E1=[abs(x) for x in RUN1[:,13]]
				VRLIST_E2=RUN1[:,14]
				VDS_E1=abs(RUN1[:,15])
				VDS_E2=abs(RUN1[:,16])
				VDS=RUN1[:,3]
				XLIST=np.zeros(len(VRLIST_E1))
				VLIST=np.zeros(len(VRLIST_E1))
				VRLIST=np.zeros(len(VRLIST_E1))
				VDS_ERR=np.zeros(len(VRLIST_E1))
				for i in range(len(VRLIST_E1)): 
					XLIST[i]=str(RUN1[i,1]) #arcsec
					VLIST[i]=str(RUN1[i,2]) #km/s
					VRLIST[i]=max(VRLIST_E1[i],VRLIST_E2[i]) #km/s
					VDS_ERR[i]=max(VDS_E1[i],VDS_E2[i]) #km/s
			VSstd=max(VSstd,abs(L_head['CDELT3']*1E-3))
			


			if dobb:
				if M0emission:

					if fitz==-1:
						finalz=(linefreq/(L_head['restFRQ']/(1+unp.uarray(VSavg,VSstd)/(cspl))))-1
						realz=(linefreq/(L_head['restFRQ']/(1+VSavg/cspl)))-1
					else:
						finalz=fitz
						realz=fitz
					print "z: "+str(finalz)
					print 'vsys: ',unp.uarray(VSavg,VSstd)
					factor=CC.getcosmos(realz,cosmoparams[0],cosmoparams[1],cosmoparams[2])[3]

			#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
			#Plot BB outputs, compare with input data
			
			if not makepanels:
				fig, axes = plt.subplots(4,3,figsize=(10,10))
				fig.suptitle(sourcefile2,fontsize=fs2,weight='bold')

			modelcube=outputfolder+sourcefile+'_MODEL.fits'
			datacube=outputfolder+sourcefile+'_L_FUL.fits'

			#~~~~~~~~~~~~~~~~~~~~
			#Make mom0 images - MOD
			m0name_im_M=modelcube.replace('.fits','m0.im')
			cubename_im_M=modelcube.replace('.fits','.im')
			mom0file_name_M=modelcube.replace('.fits','_m0_script.fits')
			ia=image()
			ia.fromfits(cubename_im_M,modelcube,overwrite=True) #Get line cube from .fits to .im
			ia.close()
			ia.open(cubename_im_M)
			subim=ia.subimage(dropdeg=True,overwrite=True)
			rg=regionmanager()
			m0im=subim.moments(moments=[0],outfile=m0name_im_M,overwrite=True)
			m0im.tofits(outfile=mom0file_name_M,overwrite=True)
			m0im.done()
			subim.done()
			ia.done()

			#Make mom1 images - MOD
			m1name_im_M=modelcube.replace('.fits','m1.im')
			cubename_im_M=modelcube.replace('.fits','.im')
			mom1file_name_M=modelcube.replace('.fits','_m1_script.fits')
			ia=image()
			ia.open(cubename_im_M)
			subim=ia.subimage(dropdeg=True,overwrite=True)
			rg=regionmanager()
			m1im=subim.moments(moments=[1],outfile=m1name_im_M,region=r,overwrite=True)
			m1im.tofits(outfile=mom1file_name_M,overwrite=True)
			m1im.done()
			subim.done()
			ia.done()

			#Make mom2 images - MOD
			m2name_im_M=modelcube.replace('.fits','m2.im')
			cubename_im_M=modelcube.replace('.fits','.im')
			mom2file_name_M=modelcube.replace('.fits','_m2_script.fits')
			ia=image()
			ia.open(cubename_im_M)
			subim=ia.subimage(dropdeg=True,overwrite=True)
			rg=regionmanager()
			m2im=subim.moments(moments=[2],outfile=m2name_im_M,region=r,overwrite=True)
			m2im.tofits(outfile=mom2file_name_M,overwrite=True)
			m2im.done()
			subim.done()
			ia.done()

			M_M012=[mom0file_name_M,mom1file_name_M,mom2file_name_M]
			D_M012=[sourcefold+sourcefile+'_m0_script_FUL.fits',sourcefold+sourcefile+'_m1_script_SIG_RED.fits',sourcefold+sourcefile+'_m2_script_SIG_RED.fits']

			#~~~~~~~~~~~~~~~~~~~~
			#Plot M0 Data
			if DATA_MASK=='DATA':
				hdu_M = fits.open(D_M012[0]) #M0_FUL
			else:
				hdu_M = fits.open(mom0file_name_S) # M0_SIG_RED
			M_data=hdu_M[0].data; M_head=fits.getheader(D_M012[0],0)
			hdu_M.close()
			temp_M_D=np.zeros((XLEN,YLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					temp_M_D[i,j]=M_data[j,i]#temp_M_D[i,j]=1E+10
			mom0_rms_D=fn.getrms_MG(temp_M_D)
			print 'Mom0 RMS:',mom0_rms
			M_BMAJ_c=M_head['BMAJ']/abs(M_head['CDELT1'])/cfact #px
			M_BMIN_c=M_head['BMIN']/abs(M_head['CDELT1'])/cfact #px
			
			#ax=plt.subplot(4,3,1,sharex='row')
			xr = range(XLEN); yr = range(YLEN)
			x, y = np.meshgrid(xr, yr)
			m0vmin=fn.matminmax(temp_M_D)[0]
			m0vmax=fn.matminmax(temp_M_D)[1]
			if not makepanels:
				axes[0,0].imshow(temp_M_D.T, cmap=cmap1, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m0vmin,vmax=m0fact*m0vmax,aspect='equal')
				if addcontours:
					axes[0,0].contour(temp_M_D.T,[X*mom0_rms for X in lev_both_2], colors='k') 
				makepretty(False,0,axes[0,0],xy0[0],xy0[1])
				xpostext=axes[0,0].get_xlim()[0]+0.05*abs(axes[0,0].get_xlim()[0]-axes[0,0].get_xlim()[1])
				ypostext=axes[0,0].get_ylim()[1]-0.13*abs(axes[0,0].get_ylim()[0]-axes[0,0].get_ylim()[1])
				if panellabels:
					axes[0,0].text(xpostext,ypostext,"M0 (Data)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes[0,0].set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
				else:
					axes[0,0].set_ylabel('Relative Dec ["]',fontsize=fs)
			else:
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				axes.imshow(temp_M_D.T, cmap=cmap1, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m0vmin,vmax=m0fact*m0vmax,aspect='equal')
				if addcontours:
					axes.contour(temp_M_D.T,[X*mom0_rms for X in lev_both_2], colors='k') 
				makepretty(False,0,axes,xy0[0],xy0[1])
				xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
				ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:				
					axes.text(xpostext,ypostext,"M0 (Data)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
					axes.set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs)
					axes.set_xlabel('Relative RA ["]',fontsize=fs)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL1.png',dpi=300,bbox_inches='tight')
				plt.close(fig)

			if addcross:
				#Make 5kpc scale
				scalex=0.80*(axes[0,0].get_xlim()[1]-axes[0,0].get_xlim()[0])+axes[0,0].get_xlim()[0]
				scaley=0.20*(axes[0,0].get_ylim()[1]-axes[0,0].get_ylim()[0])+axes[0,0].get_ylim()[0]
				scalesize=2.5/(factor*abs(M_head['cdelt1'])*3600.)
				xpos2=[scalex-scalesize,scalex+scalesize]
				ypos2=[scaley-scalesize,scaley+scalesize]
				axes[0,0].errorbar([scalex,scalex],ypos2,color='w',xerr=[1,1],zorder=3,capsize=0,lw=2)
				axes[0,0].errorbar(xpos2,[scaley,scaley],color='w',yerr=[1,1],zorder=3,capsize=0,lw=2)

			#Get model m0 into matrix
			hdu_M = fits.open(M_M012[0])
			M_data=hdu_M[0].data; M_head=fits.getheader(mom0file_name_M,0)
			hdu_M.close()
			temp_M_M=np.zeros((XLEN,YLEN))
			M0DM=fn.matminmax(M_data)[1];modelmask=np.zeros((XLEN,YLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					if normtype=='local' or (normtype=='azim' and temp_M_D[i,j]>M0DM*MODELFACTOR):
						temp_M_M[i,j]=M_data[j,i]#temp_M_M[i,j]=1E+10
						modelmask[i,j]=1
			#temp_M_M = np.ma.masked_where(temp_M_M == 1E+10, temp_M_M)
			#cmap=plt.cm.hot
			#cmap.set_bad(color='white')
			M_BMAJ_c=M_head['BMAJ']/abs(M_head['CDELT1'])/cfact #px
			M_BMIN_c=M_head['BMIN']/abs(M_head['CDELT1'])/cfact #px
			
			#Plot M0 Model
			#ax=plt.subplot(4,3,2,sharex='row')
			xr = range(XLEN); yr = range(YLEN)
			x, y = np.meshgrid(xr, yr)

			if not makepanels:
				axes[0,1].imshow(temp_M_M.T, cmap=cmap1, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m0vmin,vmax=m0fact*m0vmax,aspect='equal')
				if addcontours:
					axes[0,1].contour(temp_M_M.T,[X*mom0_rms for X in lev_both_2], colors='k') 
				makepretty(False,0,axes[0,1],xy0[0],xy0[1])
				xpostext=axes[0,1].get_xlim()[0]+0.05*abs(axes[0,1].get_xlim()[0]-axes[0,1].get_xlim()[1])
				ypostext=axes[0,1].get_ylim()[1]-0.13*abs(axes[0,1].get_ylim()[0]-axes[0,1].get_ylim()[1])
				if panellabels:
					axes[0,1].text(xpostext,ypostext,"M0 (Model)",bbox=dict(facecolor='white', alpha=1.0))
			if makepanels:
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				axes.imshow(temp_M_M.T, cmap=cmap1, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m0vmin,vmax=m0fact*m0vmax,aspect='equal')
				if addcontours:
					axes.contour(temp_M_M.T,[X*mom0_rms for X in lev_both_2], colors='k') 
				makepretty(False,0,axes,xy0[0],xy0[1])
				xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
				ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:				
					axes.text(xpostext,ypostext,"M0 (Model)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
					axes.set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs)
					axes.set_xlabel('Relative RA ["]',fontsize=fs)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL2.png',dpi=300,bbox_inches='tight')
				plt.close(fig)

			if addcross:
				#Make 5kpc scale
				scalex=0.80*(axes[0,1].get_xlim()[1]-axes[0,1].get_xlim()[0])+axes[0,1].get_xlim()[0]
				scaley=0.20*(axes[0,1].get_ylim()[1]-axes[0,1].get_ylim()[0])+axes[0,1].get_ylim()[0]
				scalesize=2.5/(factor*abs(M_head['cdelt1'])*3600.)
				xpos2=[scalex-scalesize,scalex+scalesize]
				ypos2=[scaley-scalesize,scaley+scalesize]
				axes[0,1].errorbar([scalex,scalex],ypos2,color='w',xerr=[1,1],zorder=3,capsize=0,lw=2)
				axes[0,1].errorbar(xpos2,[scaley,scaley],color='w',yerr=[1,1],zorder=3,capsize=0,lw=2)

			#Plot M0 Residual
			#Get residual m0 
			temp_m0_r=np.zeros((XLEN,YLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					temp_m0_r[i,j]=temp_M_D[i,j]-temp_M_M[i,j]

			if not makepanels:
				MP_m0_p0=axes[0,2].imshow(temp_m0_r.T, cmap=cmap1, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m0vmin,vmax=m0fact*m0vmax,aspect='equal')
				if addcontours:
					axes[0,2].contour(temp_m0_r.T,[X*mom0_rms for X in lev_both_2], colors='k') 
				makepretty(False,0,axes[0,2],xy0[0],xy0[1])
				divider = make_axes_locatable(axes[0,2])
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cbar=fig.colorbar(MP_m0_p0, use_gridspec=True, ax=axes[0,2], cax=cax)
				if nicetext:
					cbar.ax.set_ylabel('Jy/bm km/s', rotation=270,weight='bold')
				else:
					cbar.ax.set_ylabel('Jy/bm km/s', rotation=270)
				xpostext=axes[0,2].get_xlim()[0]+0.05*abs(axes[0,2].get_xlim()[0]-axes[0,2].get_xlim()[1])
				ypostext=axes[0,2].get_ylim()[1]-0.13*abs(axes[0,2].get_ylim()[0]-axes[0,2].get_ylim()[1])
				if panellabels:
					axes[0,2].text(xpostext,ypostext,"M0 (Residual)",bbox=dict(facecolor='white', alpha=1.0))
			if makepanels:
				fig, axes = plt.subplots(1,1,figsize=(5,4))
				MP_m0_p0=axes.imshow(temp_m0_r.T, cmap=cmap1, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m0vmin,vmax=m0fact*m0vmax,aspect='equal')
				if addcontours:
					axes.contour(temp_m0_r.T,[X*mom0_rms for X in lev_both_2], colors='k') 
				makepretty(False,0,axes,xy0[0],xy0[1])
				divider = make_axes_locatable(axes)
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cbar=plt.colorbar(MP_m0_p0, use_gridspec=True, ax=axes, cax=cax)
				if nicetext:
					cbar.ax.set_ylabel('Jy/bm km/s', rotation=270, weight='bold')
				else:
					cbar.ax.set_ylabel('Jy/bm km/s', rotation=270)
				xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
				ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:
					axes.text(xpostext,ypostext,"M0 (Residual)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
					axes.set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs)
					axes.set_xlabel('Relative RA ["]',fontsize=fs)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL3.png',dpi=300,bbox_inches='tight')
				plt.close(fig)

			if addcross:
				#Make 5kpc scale
				scalex=0.80*(axes[0,2].get_xlim()[1]-axes[0,2].get_xlim()[0])+axes[0,2].get_xlim()[0]
				scaley=0.20*(axes[0,2].get_ylim()[1]-axes[0,2].get_ylim()[0])+axes[0,2].get_ylim()[0]
				scalesize=2.5/(factor*abs(M_head['cdelt1'])*3600.)
				xpos2=[scalex-scalesize,scalex+scalesize]
				ypos2=[scaley-scalesize,scaley+scalesize]
				axes[0,2].errorbar([scalex,scalex],ypos2,color='w',xerr=[1,1],zorder=3,capsize=0,lw=2)
				axes[0,2].errorbar(xpos2,[scaley,scaley],color='w',yerr=[1,1],zorder=3,capsize=0,lw=2)


			#Plot M1
			hdu_M = fits.open(D_M012[1])
			M_data=hdu_M[0].data; M_head=fits.getheader(D_M012[1],0)
			hdu_M.close()
			datamask=np.zeros((XLEN,YLEN))
			temp_M_D=np.zeros((XLEN,YLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					if fn.isGood(M_data[j,i]):
						datamask[i,j]=1
					temp_M_D[i,j]=M_data[j,i]-VSavg
			M_BMAJ_c=M_head['BMAJ']/abs(M_head['CDELT1'])/cfact #px
			M_BMIN_c=M_head['BMIN']/abs(M_head['CDELT1'])/cfact #px
			#ax=plt.subplot(4,3,4)
			xr = range(XLEN); yr = range(YLEN)
			x, y = np.meshgrid(xr, yr)

			m1vBIGVAL=max(abs(fn.matminmax(temp_M_D)[0]),abs(fn.matminmax(temp_M_D)[1]))
			m1vmin=-1.*m1vBIGVAL
			m1vmax=m1vBIGVAL
			if not makepanels:
				if dogrid:
					axes[1,0].grid()
				axes[1,0].imshow(temp_M_D.T, cmap=cmap2, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m1vmin,vmax=m1vmax,aspect='equal')
				makepretty(True,PAavg,axes[1,0],xy0[0],xy0[1])
				xpostext=axes[1,0].get_xlim()[0]+0.05*abs(axes[1,0].get_xlim()[0]-axes[1,0].get_xlim()[1])
				ypostext=axes[1,0].get_ylim()[1]-0.13*abs(axes[1,0].get_ylim()[0]-axes[1,0].get_ylim()[1])
				if panellabels:
					axes[1,0].text(xpostext,ypostext,"M1 (Data)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes[1,0].set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
				else:
					axes[1,0].set_ylabel('Relative Dec ["]',fontsize=fs)
			if makepanels:
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				if dogrid:
					axes.grid()
				axes.imshow(temp_M_D.T, cmap=cmap2, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m1vmin,vmax=m1vmax,aspect='equal')
				makepretty(True,PAavg,axes,xy0[0],xy0[1])
				xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
				ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:
					axes.text(xpostext,ypostext,"M1 (Data)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
					axes.set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs)
					axes.set_xlabel('Relative RA ["]',fontsize=fs)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL4.png',dpi=300,bbox_inches='tight')
				plt.close(fig)

			#ax=plt.subplot(4,3,5)
			hdu_M = fits.open(M_M012[1])
			M_data=hdu_M[0].data; M_head=fits.getheader(M_M012[1],0)
			hdu_M.close()
			temp_M_M=np.zeros((XLEN,YLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					if modelmask[i,j]:
						temp_M_M[i,j]=M_data[j,i]-VSavg
					else:
						temp_M_M[i,j]=None
			mom1_rms_M=fn.getrms_MG(temp_M_M)
			M_BMAJ_c=M_head['BMAJ']/abs(M_head['CDELT1'])/cfact #px
			M_BMIN_c=M_head['BMIN']/abs(M_head['CDELT1'])/cfact #px
			xr = range(XLEN); yr = range(YLEN)
			x, y = np.meshgrid(xr, yr)
			if not makepanels:
				axes[1,1].imshow(temp_M_M.T, cmap=cmap2, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m1vmin,vmax=m1vmax,aspect='equal')
				makepretty(True,PAavg,axes[1,1],xy0[0],xy0[1])
				xpostext=axes[1,1].get_xlim()[0]+0.05*abs(axes[1,1].get_xlim()[0]-axes[1,1].get_xlim()[1])
				ypostext=axes[1,1].get_ylim()[1]-0.13*abs(axes[1,1].get_ylim()[0]-axes[1,1].get_ylim()[1])
				if panellabels:
					axes[1,1].text(xpostext,ypostext,"M1 (Model)",bbox=dict(facecolor='white', alpha=1.0))
				if dogrid:
					axes[1,1].grid()
			if makepanels:
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				if dogrid:
					axes.grid()
				axes.imshow(temp_M_M.T, cmap=cmap2, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m1vmin,vmax=m1vmax,aspect='equal')
				makepretty(True,PAavg,axes,xy0[0],xy0[1])
				xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
				ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:
					axes.text(xpostext,ypostext,"M1 (Model)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
					axes.set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs)
					axes.set_xlabel('Relative RA ["]',fontsize=fs)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL5.png',dpi=300,bbox_inches='tight')
				plt.close(fig)


			#Plot M1 residual
			#ax=plt.subplot(4,3,6)
			#Get residual m1 
			temp_m1_r=np.zeros((XLEN,YLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					temp_m1_r[i,j]=temp_M_D[i,j]-temp_M_M[i,j]
			if not makepanels:
				MP_m1_p0=axes[1,2].imshow(temp_m1_r.T, cmap=cmap2, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m1vmin,vmax=m1vmax,aspect='equal')
				makepretty(True,PAavg,axes[1,2],xy0[0],xy0[1])
				divider = make_axes_locatable(axes[1,2])
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cbar=plt.colorbar(MP_m1_p0, use_gridspec=True, ax=axes[1,2], cax=cax)
				if nicetext:
					cbar.ax.set_ylabel('km/s', rotation=270,weight='bold')
				else:
					cbar.ax.set_ylabel('km/s', rotation=270)
				xpostext=axes[1,2].get_xlim()[0]+0.05*abs(axes[1,2].get_xlim()[0]-axes[1,2].get_xlim()[1])
				ypostext=axes[1,2].get_ylim()[1]-0.13*abs(axes[1,2].get_ylim()[0]-axes[1,2].get_ylim()[1])
				if panellabels:				
					axes[1,2].text(xpostext,ypostext,"M1 (Residual)",bbox=dict(facecolor='white', alpha=1.0))
				if dogrid:
					axes[1,2].grid()
			if makepanels:
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				if dogrid:
					axes.grid()
				MP_m1_p0=axes.imshow(temp_m1_r.T, cmap=cmap2, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m1vmin,vmax=m1vmax,aspect='equal')
				makepretty(True,PAavg,axes,xy0[0],xy0[1])
				divider = make_axes_locatable(axes)
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cbar=plt.colorbar(MP_m1_p0, use_gridspec=True, ax=axes, cax=cax)
				if nicetext:
					cbar.ax.set_ylabel('km/s', rotation=270,weight='bold')
				else:
					cbar.ax.set_ylabel('km/s', rotation=270)
				xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
				ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:				
					axes.text(xpostext,ypostext,"M1 (Residual)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
					axes.set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs)
					axes.set_xlabel('Relative RA ["]',fontsize=fs)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL6.png',dpi=300,bbox_inches='tight')
				plt.close(fig)


			hdu_M = fits.open(D_M012[2])
			M_data=hdu_M[0].data; M_head=fits.getheader(D_M012[2],0)
			hdu_M.close()
			temp_M_D=np.zeros((XLEN,YLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					temp_M_D[i,j]=M_data[j,i]
			mom2_rms_D=fn.getrms_MG(temp_M_D)
			M_BMAJ_c=M_head['BMAJ']/abs(M_head['CDELT1'])/cfact #px
			M_BMIN_c=M_head['BMIN']/abs(M_head['CDELT1'])/cfact #px

			#Plot M2
			#ax=plt.subplot(4,3,7)
			xr = range(XLEN); yr = range(YLEN)
			x, y = np.meshgrid(xr, yr)
			m2vmin=fn.matminmax(temp_M_D)[0]
			m2vmax=fn.matminmax(temp_M_D)[1]
			if not makepanels:
				axes[2,0].imshow(temp_M_D.T, cmap=cmap3, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m2vmin,vmax=m2vmax,aspect='equal')
				if dogrid:
					axes[2,0].grid()
				makepretty(False,0,axes[2,0],xy0[0],xy0[1])
				xpostext=axes[2,0].get_xlim()[0]+0.05*abs(axes[2,0].get_xlim()[0]-axes[2,0].get_xlim()[1])
				ypostext=axes[2,0].get_ylim()[1]-0.13*abs(axes[2,0].get_ylim()[0]-axes[2,0].get_ylim()[1])
				if panellabels:
					axes[2,0].text(xpostext,ypostext,"M2 (Data)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes[2,0].set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
					axes[2,0].set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes[2,0].set_ylabel('Relative Dec ["]',fontsize=fs)
					axes[2,0].set_xlabel('Relative RA ["]',fontsize=fs)

			if makepanels:
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				if dogrid:
					axes.grid()
				axes.imshow(temp_M_D.T, cmap=cmap3, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m2vmin,vmax=m2vmax,aspect='equal')
				makepretty(False,0,axes,xy0[0],xy0[1])
				xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
				ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:
					axes.text(xpostext,ypostext,"M2 (Data)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
					axes.set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs)
					axes.set_xlabel('Relative RA ["]',fontsize=fs)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL7.png',dpi=300,bbox_inches='tight')					
				plt.close(fig)

			hdu_M = fits.open(M_M012[2])
			M_data=hdu_M[0].data; M_head=fits.getheader(M_M012[2],0)
			hdu_M.close()
			temp_M_M=np.zeros((XLEN,YLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					if modelmask[i,j]:
						temp_M_M[i,j]=M_data[j,i]
					else:
						temp_M_M[i,j]=None
			mom2_rms_M=fn.getrms_MG(temp_M_M)
			M_BMAJ_c=M_head['BMAJ']/abs(M_head['CDELT1'])/cfact #px
			M_BMIN_c=M_head['BMIN']/abs(M_head['CDELT1'])/cfact #px
			
			#Plot M2
			#ax=plt.subplot(4,3,8)
			xr = range(XLEN); yr = range(YLEN)
			x, y = np.meshgrid(xr, yr)
			if not makepanels:
				axes[2,1].imshow(temp_M_M.T, cmap=cmap3, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m2vmin,vmax=m2vmax,aspect='equal')
				makepretty(False,0,axes[2,1],xy0[0],xy0[1])
				xpostext=axes[2,1].get_xlim()[0]+0.05*abs(axes[2,1].get_xlim()[0]-axes[2,1].get_xlim()[1])
				ypostext=axes[2,1].get_ylim()[1]-0.13*abs(axes[2,1].get_ylim()[0]-axes[2,1].get_ylim()[1])
				if panellabels:
					axes[2,1].text(xpostext,ypostext,"M2 (Model)",bbox=dict(facecolor='white', alpha=1.0))
				if dogrid:
					axes[2,1].grid()
				if nicetext:
					axes[2,1].set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes[2,1].set_xlabel('Relative RA ["]',fontsize=fs)
			if makepanels:
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				if dogrid:
					axes.grid()
				axes.imshow(temp_M_M.T, cmap=cmap3, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m2vmin,vmax=m2vmax,aspect='equal')
				makepretty(False,0,axes,xy0[0],xy0[1])
				xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
				ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:
					axes.text(xpostext,ypostext,"M2 (Model)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
					axes.set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs)
					axes.set_xlabel('Relative RA ["]',fontsize=fs)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL8.png',dpi=300,bbox_inches='tight')
				plt.close(fig)

			#ax=plt.subplot(4,3,9)
			#Get residual m2 
			temp_m2_r=np.zeros((XLEN,YLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					temp_m2_r[i,j]=abs(temp_M_D[i,j]-temp_M_M[i,j])
			if not makepanels:
				MP_m2_p0=axes[2,2].imshow(temp_m2_r.T, cmap=cmap3, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m2vmin,vmax=m2vmax,aspect='equal')
				makepretty(False,0,axes[2,2],xy0[0],xy0[1])
				divider = make_axes_locatable(axes[2,2])
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cb2=plt.colorbar(MP_m2_p0, use_gridspec=True, ax=axes[2,2], cax=cax)
				if nicetext:
					cb2.ax.set_ylabel('km/s', rotation=270, weight='bold')
				else:
					cb2.ax.set_ylabel('km/s', rotation=270)
				xpostext=axes[2,2].get_xlim()[0]+0.05*abs(axes[2,2].get_xlim()[0]-axes[2,2].get_xlim()[1])
				ypostext=axes[2,2].get_ylim()[1]-0.13*abs(axes[2,2].get_ylim()[0]-axes[2,2].get_ylim()[1])
				if panellabels:
					axes[2,2].text(xpostext,ypostext,"M2 (Residual)",bbox=dict(facecolor='white', alpha=1.0))
				if dogrid:
					axes[2,2].grid()
				if nicetext:
					axes[2,2].set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes[2,2].set_xlabel('Relative RA ["]',fontsize=fs)
			if makepanels:
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				if dogrid:
					axes.grid()
				MP_m2_p0=axes.imshow(temp_m2_r.T, cmap=cmap3, origin='bottom',extent=(x.min(), x.max(), y.min(), y.max()),interpolation='nearest',vmin=m2vmin,vmax=m2vmax,aspect='equal')
				makepretty(False,0,axes,xy0[0],xy0[1])
				divider = make_axes_locatable(axes)
				cax = divider.append_axes("right", size="5%", pad=0.05)
				cb2=plt.colorbar(MP_m2_p0, use_gridspec=True, ax=axes, cax=cax)
				if nicetext:
					cb2.ax.set_ylabel('km/s', rotation=270, weight='bold')
				else:
					cb2.ax.set_ylabel('km/s', rotation=270)
				xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
				ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:
					axes.text(xpostext,ypostext,"M2 (Residual)",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs,weight='bold')
					axes.set_xlabel('Relative RA ["]',fontsize=fs,weight='bold')
				else:
					axes.set_ylabel('Relative Dec ["]',fontsize=fs)
					axes.set_xlabel('Relative RA ["]',fontsize=fs)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL9.png',dpi=300,bbox_inches='tight')	
				plt.close(fig)
			
			#============================================
			#ROW 4 ======================================
			#============================================

			print 'PV...'
			
			#PV of data
			truecenter=[xy0[0],xy0[1]]
			cubename_im=outputfolder+sourcefile+'_L_FUL.fits'
			ia=image()
			ia.open(cubename_im)
			print 'Making PV of',cubename_im,'...'
			mypv = ia.pv(outfile="pv1.im", center=truecenter, length=length, pa=str(PAavg)+'deg', width=3, overwrite=True, wantreturn=true)  
			mypv.tofits(outfile=PV1file_name,overwrite=True)
			mypv.done()
			mypv = ia.pv(outfile="pv2.im", center=truecenter, length=length, pa=str(PAavg+90.)+'deg', width=3, overwrite=True, wantreturn=true)  
			mypv.tofits(outfile=PV2file_name,overwrite=True)
			mypv.done()
			ia.done()
			#PV of model
			cubename_im_M=outputfolder+sourcefile+'_MODEL.fits'
			ia=image()
			ia.open(cubename_im_M)
			print 'Making PV of',cubename_im_M,'...'
			mypv = ia.pv(outfile="pv1_M.im", center=truecenter, length=length, pa=str(PAavg)+'deg', width=3, overwrite=True, wantreturn=true)  
			mypv.tofits(outfile=PV1file_name.replace('PV1','PV1_M'),overwrite=True)
			mypv.done()
			mypv = ia.pv(outfile="pv2_M.im", center=truecenter, length=length, pa=str(PAavg+90.)+'deg', width=3, overwrite=True, wantreturn=true)  
			mypv.tofits(outfile=PV2file_name.replace('PV2','PV2_M'),overwrite=True)
			mypv.done()
			ia.done()

			#============================================
			#ROW 4 COLUMN 1
			#============================================

			if delv2==-1:
				delv2=int(delv/2)

			#Get PV1 into matrix
			hdu_PV1 = fits.open(PV1file_name)
			PV1_data=hdu_PV1[0].data; PV1_head=fits.getheader(PV1file_name,0)
			hdu_PV1.close()
			hdu_PV1_M = fits.open(PV1file_name.replace('PV1','PV1_M'))
			PV1_data_M=hdu_PV1_M[0].data; PV1_head_M=fits.getheader(PV1file_name.replace('PV1','PV1_M'),0)
			hdu_PV1_M.close()
			'''
			for pv_i in range(PV1_data.shape[0]): 
				for pv_j in range(PV1_data.shape[1]):
					if not fn.isGood(PV1_data[pv_i,pv_j]):
						PV1_data[pv_i,pv_j]=0
					if not fn.isGood(PV1_data_M[pv_i,pv_j]):
						PV1_data_M[pv_i,pv_j]=0
			'''
			if not makepanels:
				axes[3,0].imshow(PV1_data, cmap=cmap4, origin='bottom',interpolation='nearest',aspect='equal')
				#Figure out PV contours
				tempmx=-1E+99
				for i in range(PV1_data_M.shape[0]):
					for j in range(PV1_data_M.shape[1]):
						if PV1_data_M[i,j]>tempmx: tempmx=PV1_data_M[i,j]
				lev_both=[0.2,0.5,0.8]
				axes[3,0].contour(PV1_data_M,[X*tempmx for X in lev_both], colors='k') 
				superdupertemp=range(-10,11)
				xlist1_PV=[zoomnum*x for x in superdupertemp]; xlist2_PV=np.zeros(len(xlist1_PV))
				ylist1_PV=range(-10*delv2,11*delv2,delv2); ylist2_PV=np.zeros(len(ylist1_PV))
				for i in range(len(xlist1_PV)):
					xlist2_PV[i]=-1+PV1_head['CRPIX1']+((xlist1_PV[i]-PV1_head['CRVAL1'])/PV1_head['CDELT1'])
					xlist1_PV[i]=round(xlist1_PV[i],1)
				for i in range(len(ylist1_PV)):
					ylist2_PV[i]=-1+PV1_head['CRPIX2']+(((RF/(1+((VSavg+ylist1_PV[i])/cspl)))-PV1_head['CRVAL2'])/PV1_head['CDELT2'])
					#ylist2_PV[i]=-1+PV1_head['CRPIX2']+(((RF/(1+((ylist1_PV[i]+VSavg)/cspl)))-PV1_head['CRVAL2'])/PV1_head['CDELT2'])
				axes[3,0].set_xticks(xlist2_PV)
				axes[3,0].set_xticklabels(xlist1_PV)#,rotation=45)
				axes[3,0].set_yticks(ylist2_PV)
				axes[3,0].set_yticklabels(ylist1_PV)
				axes[3,0].set_xlim(0,PV1_data.shape[1]-1)
				if not upsidedown:
					axes[3,0].set_ylim(max(0,m0_ch_low-5),min(m0_ch_high+5,PV1_data.shape[0]-1))
					xpostext=axes[3,0].get_xlim()[0]+0.05*abs(axes[3,0].get_xlim()[0]-axes[3,0].get_xlim()[1])
					if not PVR:
						ypostext=axes[3,0].get_ylim()[1]-0.13*abs(axes[3,0].get_ylim()[0]-axes[3,0].get_ylim()[1])
					else:
						ypostext=axes[3,0].get_ylim()[0]+0.08*abs(axes[3,0].get_ylim()[0]-axes[3,0].get_ylim()[1])
				else:
					axes[3,0].set_ylim(min(m0_ch_high+5,PV1_data.shape[0]-1),max(0,m0_ch_low-5))
					xpostext=axes[3,0].get_xlim()[0]+0.05*abs(axes[3,0].get_xlim()[0]-axes[3,0].get_xlim()[1])
					if not PVR:
						ypostext=axes[3,0].get_ylim()[1]+0.13*abs(axes[3,0].get_ylim()[0]-axes[3,0].get_ylim()[1])
					else:
						ypostext=axes[3,0].get_ylim()[1]-0.13*abs(axes[3,0].get_ylim()[0]-axes[3,0].get_ylim()[1])
				if panellabels:
					axes[3,0].text(xpostext,ypostext,"Major Axis",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes[3,0].set_xlabel('Offset [\"]',fontsize=fs,weight='bold')
					axes[3,0].set_ylabel('Velocity [km/s]',fontsize=fs,weight='bold')
				else:
					axes[3,0].set_xlabel('Offset [\"]',fontsize=fs)
					axes[3,0].set_ylabel('Velocity [km/s]',fontsize=fs)
				aspectnum=(max(axes[3,0].get_ylim())-min(axes[3,0].get_ylim()))/(max(axes[3,0].get_xlim())-min(axes[3,0].get_xlim()))
				axes[3,0].set_aspect(1.0/aspectnum, adjustable='box', anchor='C') #auto is eh. equal is worse.
			if makepanels:
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				PVvmin=fn.matminmax(PV1_data)[0]
				PVvmax=fn.matminmax(PV1_data)[1]
				axes.imshow(PV1_data, cmap=cmap4, origin='bottom',interpolation='nearest',aspect='equal',zorder=0,vmin=PVvmin,vmax=PVvmax)
				tempmx=-1E+99
				for i in range(PV1_data_M.shape[0]):
					for j in range(PV1_data_M.shape[1]):
						if PV1_data_M[i,j]>tempmx: tempmx=PV1_data_M[i,j]
				lev_both=[0.2,0.5,0.8]
				axes.contour(PV1_data_M,[X*tempmx for X in lev_both], colors='k',zorder=10) 
				xlist1_PV=[zoomnum*x for x in superdupertemp]; xlist2_PV=np.zeros(len(xlist1_PV))
				ylist1_PV=range(-10*delv2,11*delv2,delv2); ylist2_PV=np.zeros(len(ylist1_PV))
				for i in range(len(xlist1_PV)):
					xlist2_PV[i]=-1+PV1_head['CRPIX1']+((xlist1_PV[i]-PV1_head['CRVAL1'])/PV1_head['CDELT1'])
					xlist1_PV[i]=round(xlist1_PV[i],1)
				for i in range(len(ylist1_PV)):
					ylist2_PV[i]=-1+PV1_head['CRPIX2']+(((RF/(1+((VSavg+ylist1_PV[i])/cspl)))-PV1_head['CRVAL2'])/PV1_head['CDELT2'])
				axes.set_xticks(xlist2_PV)
				axes.set_xticklabels(xlist1_PV)#,rotation=45)
				axes.set_yticks(ylist2_PV)
				axes.set_yticklabels(ylist1_PV)
				axes.set_xlim(0,PV1_data.shape[1]-1)
				if not upsidedown:
					axes.set_ylim(max(0,m0_ch_low-5),min(m0_ch_high+5,PV1_data.shape[0]-1))
					xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
					if not PVR:
						ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
					else:
						ypostext=axes.get_ylim()[0]+0.08*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				else:
					axes.set_ylim(min(m0_ch_high+5,PV1_data.shape[0]-1),max(0,m0_ch_low-5))
					xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
					if not PVR:
						ypostext=axes.get_ylim()[1]+0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
					else:
						ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:
					axes.text(xpostext,ypostext,"Major Axis",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_xlabel('Offset [\"]',fontsize=fs,weight='bold')
					axes.set_ylabel('Velocity [km/s]',fontsize=fs,weight='bold')
				else:
					axes.set_xlabel('Offset [\"]',fontsize=fs)
					axes.set_ylabel('Velocity [km/s]',fontsize=fs)
				aspectnum=(max(axes.get_ylim())-min(axes.get_ylim()))/(max(axes.get_xlim())-min(axes.get_xlim()))
				axes.set_aspect(1.0/aspectnum, adjustable='box', anchor='C') #auto is eh. equal is worse.
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL10.png',dpi=300,bbox_inches='tight')			
				#
				tempPV=np.zeros((PV1_data.shape[0],PV1_data.shape[1]))
				for pvi in range(PV1_data.shape[0]):
					for pvj in range(PV1_data.shape[1]):
						tempPV[pvi,pvj]=PV1_data[pvi,pvj]-PV1_data_M[pvi,pvj]
				axes.imshow(tempPV, cmap=cmap4, origin='bottom',interpolation='nearest',aspect='equal',zorder=100,vmin=PVvmin,vmax=PVvmax)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL10a.png',dpi=300,bbox_inches='tight')
				plt.close(fig)	

			#============================================
			#ROW 4 COLUMN 2
			#============================================

			#Get PV2 into matrix
			hdu_PV2 = fits.open(PV2file_name)
			PV2_data=hdu_PV2[0].data; PV2_head=fits.getheader(PV2file_name,0)
			hdu_PV2.close()
			hdu_PV2_M = fits.open(PV2file_name.replace('PV2','PV2_M'))
			PV2_data_M=hdu_PV2_M[0].data; PV2_head_M=fits.getheader(PV2file_name.replace('PV2','PV2_M'),0)
			hdu_PV2_M.close()
			'''
			for pv_i in range(PV2_data.shape[0]): 
				for pv_j in range(PV2_data.shape[1]):
					if not fn.isGood(PV2_data[pv_i,pv_j]):
						PV2_data[pv_i,pv_j]=0
					if not fn.isGood(PV2_data_M[pv_i,pv_j]):
						PV2_data_M[pv_i,pv_j]=0
			'''
			if not makepanels:
				axes[3,1].imshow(PV2_data, cmap=cmap4, origin='bottom',interpolation='nearest',aspect='equal')
				#Figure out PV contours
				tempmx=-1E+99
				for i in range(PV2_data_M.shape[0]):
					for j in range(PV2_data_M.shape[1]):
						if PV2_data_M[i,j]>tempmx: tempmx=PV2_data_M[i,j]
				lev_both=[0.2,0.5,0.8]
				axes[3,1].contour(PV2_data_M,[X*tempmx for X in lev_both], colors='k') 
				xlist1_PV=[zoomnum*x for x in superdupertemp]; xlist2_PV=np.zeros(len(xlist1_PV))
				ylist1_PV=range(-10*delv2,11*delv2,delv2); ylist2_PV=np.zeros(len(ylist1_PV))
				for i in range(len(xlist1_PV)):
					xlist1_PV[i]=round(xlist1_PV[i],1)	
					xlist2_PV[i]=-1+PV2_head['CRPIX1']+((xlist1_PV[i]-PV2_head['CRVAL1'])/PV2_head['CDELT1'])
				for i in range(len(ylist1_PV)):
					ylist2_PV[i]=-1+PV2_head['CRPIX2']+(((RF/(1+((VSavg+ylist1_PV[i])/cspl)))-PV2_head['CRVAL2'])/PV2_head['CDELT2'])
					#ylist2_PV[i]=-1+PV2_head['CRPIX2']+(((RF/(1+((ylist1_PV[i]+VSavg)/cspl)))-PV2_head['CRVAL2'])/PV2_head['CDELT2'])
				axes[3,1].set_xticks(xlist2_PV)
				axes[3,1].set_xticklabels(xlist1_PV)#,rotation=45)
				axes[3,1].set_yticks(ylist2_PV)
				axes[3,1].set_yticklabels(ylist1_PV)
				axes[3,1].set_xlim(0,PV2_data.shape[1]-1)
				if not upsidedown:
					axes[3,1].set_ylim(max(0,m0_ch_low-5),min(m0_ch_high+5,PV2_data.shape[0]-1))
					xpostext=axes[3,1].get_xlim()[0]+0.05*abs(axes[3,1].get_xlim()[0]-axes[3,1].get_xlim()[1])
					if not PVR:
						ypostext=axes[3,1].get_ylim()[1]-0.13*abs(axes[3,1].get_ylim()[0]-axes[3,1].get_ylim()[1])
					else:
						ypostext=axes[3,1].get_ylim()[0]+0.08*abs(axes[3,1].get_ylim()[0]-axes[3,1].get_ylim()[1])
				else:
					axes[3,1].set_ylim(min(m0_ch_high+5,PV2_data.shape[0]-1),max(0,m0_ch_low-5))
					xpostext=axes[3,1].get_xlim()[0]+0.05*abs(axes[3,1].get_xlim()[0]-axes[3,1].get_xlim()[1])
					if not PVR:
						ypostext=axes[3,1].get_ylim()[1]+0.13*abs(axes[3,1].get_ylim()[0]-axes[3,1].get_ylim()[1])
					else:
						ypostext=axes[3,1].get_ylim()[1]-0.13*abs(axes[3,1].get_ylim()[0]-axes[3,1].get_ylim()[1])
				if panellabels:
					axes[3,1].text(xpostext,ypostext,"Minor Axis",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes[3,1].set_xlabel('Offset [\"]',fontsize=fs,weight='bold')
				else:
					axes[3,1].set_xlabel('Offset [\"]',fontsize=fs)	
				aspectnum=(max(axes[3,1].get_ylim())-min(axes[3,1].get_ylim()))/(max(axes[3,1].get_xlim())-min(axes[3,1].get_xlim()))
				axes[3,1].set_aspect(1.0/aspectnum, adjustable='box', anchor='C') #auto is eh. equal is worse.
			if makepanels:
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				PVvmin=fn.matminmax(PV2_data)[0]
				PVvmax=fn.matminmax(PV2_data)[1]
				axes.imshow(PV2_data, cmap=cmap4, origin='bottom',interpolation='nearest',aspect='equal',zorder=0,vmin=PVvmin,vmax=PVvmax)
				tempmx=-1E+99
				for i in range(PV2_data_M.shape[0]):
					for j in range(PV2_data_M.shape[1]):
						if PV2_data_M[i,j]>tempmx: tempmx=PV2_data_M[i,j]
				lev_both=[0.2,0.5,0.8]
				axes.contour(PV2_data_M,[X*tempmx for X in lev_both], colors='k') 
				xlist1_PV=[zoomnum*x for x in superdupertemp]; xlist2_PV=np.zeros(len(xlist1_PV))
				ylist1_PV=range(-10*delv2,11*delv2,delv2); ylist2_PV=np.zeros(len(ylist1_PV))
				for i in range(len(xlist1_PV)):
					xlist2_PV[i]=-1+PV2_head['CRPIX1']+((xlist1_PV[i]-PV2_head['CRVAL1'])/PV2_head['CDELT1'])
					xlist1_PV[i]=round(xlist1_PV[i],1)
				for i in range(len(ylist1_PV)):
					ylist2_PV[i]=-1+PV2_head['CRPIX2']+(((RF/(1+((VSavg+ylist1_PV[i])/cspl)))-PV2_head['CRVAL2'])/PV2_head['CDELT2'])
					#ylist2_PV[i]=-1+PV2_head['CRPIX2']+(((RF/(1+((ylist1_PV[i]+VSavg)/cspl)))-PV2_head['CRVAL2'])/PV2_head['CDELT2'])
				axes.set_xticks(xlist2_PV)
				axes.set_xticklabels(xlist1_PV)#,rotation=45)
				axes.set_yticks(ylist2_PV)
				axes.set_yticklabels(ylist1_PV)
				axes.set_xlim(0,PV2_data.shape[1]-1)
				if not upsidedown:
					axes.set_ylim(max(0,m0_ch_low-5),min(m0_ch_high+5,PV2_data.shape[0]-1))
					xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
					if not PVR:
						ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
					else:
						ypostext=axes.get_ylim()[0]+0.08*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				else:
					axes.set_ylim(min(m0_ch_high+5,PV2_data.shape[0]-1),max(0,m0_ch_low-5))
					xpostext=axes.get_xlim()[0]+0.05*abs(axes.get_xlim()[0]-axes.get_xlim()[1])
					if not PVR:
						ypostext=axes.get_ylim()[1]+0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
					else:
						ypostext=axes.get_ylim()[1]-0.13*abs(axes.get_ylim()[0]-axes.get_ylim()[1])
				if panellabels:
					axes.text(xpostext,ypostext,"Minor Axis",bbox=dict(facecolor='white', alpha=1.0))
				if nicetext:
					axes.set_xlabel('Offset [\"]',fontsize=fs,weight='bold')
					axes.set_ylabel('Velocity [km/s]',fontsize=fs,weight='bold')
				else:
					axes.set_xlabel('Offset [\"]',fontsize=fs)
					axes.set_ylabel('Velocity [km/s]',fontsize=fs)
				aspectnum=(max(axes.get_ylim())-min(axes.get_ylim()))/(max(axes.get_xlim())-min(axes.get_xlim()))
				axes.set_aspect(1.0/aspectnum, adjustable='box', anchor='C') #auto is eh. equal is worse.
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL11.png',dpi=300,bbox_inches='tight')	
				#
				tempPV=np.zeros((PV2_data.shape[0],PV2_data.shape[1]))
				for pvi in range(PV2_data.shape[0]):
					for pvj in range(PV2_data.shape[1]):
						tempPV[pvi,pvj]=PV2_data[pvi,pvj]-PV2_data_M[pvi,pvj]
				axes.imshow(tempPV, cmap=cmap4, origin='bottom',interpolation='nearest',aspect='equal',zorder=100,vmin=PVvmin,vmax=PVvmax)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL11a.png',dpi=300,bbox_inches='tight')
				plt.close(fig)	



			#============================================
			#ROW 4 COLUMN 3
			#============================================

			print 'Spectrum...'
			#datacube=sourcefold+sourcefile+'.fits'

			if DATA_MASK=='DATA':
				datacube=outputfolder+sourcefile+'_L_FUL.fits'
			else:
				datacube=outputfolder+sourcefile+'_L_SIG_RED2.fits'
			hdu_L = fits.open(datacube)
			DAT_data=hdu_L[0].data; DAT_head=fits.getheader(datacube,0)
			hdu_L.close()
			temp_DAT=np.zeros((XLEN,YLEN,ZLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					for k in range(ZLEN):
						if fn.isGood(DAT_data[k,j,i]): 
							temp_DAT[i,j,k]=DAT_data[k,j,i]
			#
			hdu_L = fits.open(modelcube)
			MOD_data=hdu_L[0].data; MOD_head=fits.getheader(modelcube,0)
			hdu_L.close()
			temp_MOD=np.zeros((XLEN,YLEN,ZLEN))
			for i in range(XLEN):
				for j in range(YLEN):
					for k in range(ZLEN):
						if fn.isGood(MOD_data[k,j,i]): 
							temp_MOD[i,j,k]=MOD_data[k,j,i]

			#Get data and model spectra
			bigspec_M=np.zeros(ZLEN)
			bigspec_D=np.zeros(ZLEN)
			bigspec_RMS=np.zeros(ZLEN)
			datacount=np.zeros(ZLEN)
			
			for i in range(XLEN):
				for j in range(YLEN):
					for k in range(ZLEN):
						bigspec_M[k]+=(1E+3*temp_MOD[i,j,k])
					if datamask[i,j]==1:#1==1:#temp_MOD[k,j,i]>0.:#datamask[k,j]:#sigmask[k,j]
						for k in range(ZLEN):
							bigspec_D[k]+=(1E+3*temp_DAT[i,j,k])
							datacount[k]+=1
			for k in range(ZLEN):
				bigspec_RMS[k]=(1E+3)*(chrms*np.sqrt(datacount[k]/beampx))
			specname=sourcefold+sourcefile+'_spectrum.txt'
			try:
				f=open(specname,'w');f.close()
			except IOError:
				pass
			f=open(specname,'w')
			for i in range(ZLEN):
				f.write(str(fn.CHtoV(L_head,i))+' '+str(bigspec_D[i]/beampx)+' '+str(bigspec_M[i]/beampx)+'\n')
			f.close()
			if not makepanels:
				axes[3,2].step(range(ZLEN),bigspec_D/beampx,label='D',c=threecolors[0],lw=linw)
				axes[3,2].step(range(ZLEN),bigspec_M/beampx,label='M',c=threecolors[1],lw=linw)
				axes[3,2].step(range(ZLEN),(bigspec_D-bigspec_M)/beampx,label='R',c=threecolors[2],lw=linw)
				axes[3,2].fill_between(range(ZLEN),bigspec_RMS,color='k',alpha=0.2)
				axes[3,2].fill_between(range(ZLEN),-1*bigspec_RMS,color='k',alpha=0.2)
				if nicetext:
					axes[3,2].set_xlabel('Velocity [km/s]',fontsize=fs,weight='bold')
					axes[3,2].set_ylabel('Flux Density [mJy]',fontsize=fs,weight='bold')
				else:
					axes[3,2].set_xlabel('Velocity [km/s]',fontsize=fs)
					axes[3,2].set_ylabel('Flux Density [mJy]',fontsize=fs)
				xlist1=range(-10*delv,11*delv,delv)
				xlist2=np.zeros(len(xlist1))
				if L_head['CUNIT3']=='m/s':
					for i in range(len(xlist1)):
						xlist2[i]=-1+(L_head['CRPIX3'])+((xlist1[i]+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3']))
					axes[3,2].axvline(-1+(L_head['CRPIX3'])+((0.+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3'])),linestyle='dashed',c='k')
					if float(L_head['CDELT3'])<0:
						axes[3,2].set_xticks(xlist2)
						axes[3,2].set_xticklabels(xlist1)
						if bbxlim:
							bbxlimA=-1+(L_head['CRPIX3'])+((-750.+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3']))
							bbxlimB=-1+(L_head['CRPIX3'])+((750.+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3']))
							axes[3,2].set_xlim(bbxlimA,bbxlimB)
						elif bbxlim2:
							bbxlimA=-1+(L_head['CRPIX3'])+((-1000.+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3']))
							bbxlimB=-1+(L_head['CRPIX3'])+((1000.+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3']))
							axes[3,2].set_xlim(bbxlimA,bbxlimB)
						else:
							axes[3,2].set_xlim(ZLEN-1,0)
					else:
						axes[3,2].set_xticks(xlist2)
						axes[3,2].set_xticklabels(xlist1)
						if bbxlim:
							bbxlimA=-1+(L_head['CRPIX3'])+((-750.+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3']))
							bbxlimB=-1+(L_head['CRPIX3'])+((750.+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3']))
							axes[3,2].set_xlim(bbxlimA,bbxlimB)
						elif bbxlim2:
							bbxlimA=-1+(L_head['CRPIX3'])+((-1000.+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3']))
							bbxlimB=-1+(L_head['CRPIX3'])+((1000.+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3']))
							axes[3,2].set_xlim(bbxlimA,bbxlimB)
						else:
							axes[3,2].set_xlim(0,ZLEN-1)
						
				else:
					print 'BAD CUNIT3'
				BIGMIN=min(min(min(bigspec_D/beampx),min(bigspec_M/beampx)),min((bigspec_D-bigspec_M)/beampx))
				BIGMAX=max(max(max(bigspec_D/beampx),max(bigspec_M/beampx)),max((bigspec_D-bigspec_M)/beampx))
				axes[3,2].set_ylim(1.2*BIGMIN,1.2*BIGMAX)
				axes[3,2].tick_params(axis='x', which='minor', bottom=True)
				axes[3,2].legend(loc=LR,prop={'size': 10})
				axes[3,2].axhline(y=0,linestyle='solid',c='k',lw=linw)
				aspectnum=(max(axes[3,2].get_ylim())-min(axes[3,2].get_ylim()))/(max(axes[3,2].get_xlim())-min(axes[3,2].get_xlim()))
				axes[3,2].set_aspect(aspect='auto')#0.5/aspectnum, adjustable='box', anchor='E') #auto is eh. equal is worse.
				plt.tight_layout(pad=3.0,h_pad=0.5,w_pad=0.5)
				fig.subplots_adjust(top=0.925)
				tempname=sourcefold+sourcefile.replace('.','_')
				print 'Saving',tempname
				plt.savefig(tempname+'bigplot.png',dpi=300,bbox_inches='tight')
				#plt.close(fig)
			if makepanels:	
				fig, axes = plt.subplots(1,1,figsize=(4,4))
				axes.step(range(ZLEN),bigspec_M/beampx,label='M',c='r',lw=linw)
				axes.step(range(ZLEN),bigspec_D/beampx,label='D',c='b',lw=linw)
				axes.step(range(ZLEN),(bigspec_D-bigspec_M)/beampx,label='R',c='g',lw=linw)
				axes.fill_between(range(ZLEN),bigspec_RMS,color='k',alpha=0.2)
				axes.fill_between(range(ZLEN),-1*bigspec_RMS,color='k',alpha=0.2)
				if nicetext:
					axes.set_xlabel('Velocity [km/s]',fontsize=fs,weight='bold')
					axes.set_ylabel('Flux Density [mJy]',fontsize=fs,weight='bold')
				else:
					axes.set_xlabel('Velocity [km/s]',fontsize=fs)
					axes.set_ylabel('Flux Density [mJy]',fontsize=fs)
				xlist1=range(-10*delv,11*delv,delv)
				xlist2=np.zeros(len(xlist1))
				if L_head['CUNIT3']=='m/s':
					for i in range(len(xlist1)):
						xlist2[i]=-1+(L_head['CRPIX3'])+((xlist1[i]+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3']))
					axes.axvline(-1+(L_head['CRPIX3'])+((0.+VSavg-(1E-3*L_head['CRVAL3']))/(1E-3*L_head['CDELT3'])),linestyle='dashed',c='k')
					if float(L_head['CDELT3'])<0:
						axes.set_xticks(xlist2)
						axes.set_xticklabels(xlist1)
						axes.set_xlim(ZLEN-1,0)
					else:
						axes.set_xticks(xlist2)
						axes.set_xticklabels(xlist1)
						axes.set_xlim(0,ZLEN-1)
				else:
					print 'BAD CUNIT3'
				BIGMIN=min(min(min(bigspec_D/beampx),min(bigspec_M/beampx)),min((bigspec_D-bigspec_M)/beampx))
				BIGMAX=max(max(max(bigspec_D/beampx),max(bigspec_M/beampx)),max((bigspec_D-bigspec_M)/beampx))
				axes.set_ylim(1.2*BIGMIN,1.2*BIGMAX)
				axes.tick_params(axis='x', which='minor', bottom=True)
				axes.legend(loc=LR,prop={'size': 10})
				axes.axhline(y=0,linestyle='solid',c='k',lw=linw)
				aspectnum=(max(axes.get_ylim())-min(axes.get_ylim()))/(max(axes.get_xlim())-min(axes.get_xlim()))
				axes.set_aspect(0.7/aspectnum, adjustable='box', anchor='E') #auto is eh. equal is worse.
				#plt.tight_layout(pad=3.0,h_pad=0.5,w_pad=0.5)
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'PANEL12.png',dpi=300,bbox_inches='tight')
				plt.close(fig)


	else:
		print 'WHY???'
	
	#~~~~~~~~~~~
	#Print out everything
	print "Average Channel RMS: "+str(1000.*chrms)+" [mJy/bm]"
	if mom0success:
		print "Peak: "+str(M_Ampl)+" [Jy/bm km/s]"
		print "ra: "+str(fn.convra(M_RA_rad*180./np.pi))+" +/- "+str(round(M_RA_as_e/15.,3))+"s ("+str(round(M_RA_as_e,2))+" arcsec)" #WRONG ERROR
		print "dec: "+str(fn.convdec(M_DEC_rad*180./np.pi))+" +/- "+str(round(M_DEC_as_e,2))+" arcsec" #WRONG ERROR
		print "(x,y): ("+str(M_x0)+" , "+str(M_y0)+")"
		print "Integrated: "+str(M_int)+" [mJy]"
		if M_fwhmx<1E+10:
			print "FWHM_maj: "+str(M_fwhmx)+" [arcsec]"
			print "FWHM_min: "+str(M_fwhmy)+" [arcsec]"
			#print "PA: "+str(M_theta)+" [degrees]"
		else: 
			print 'Not Resolved'
	else:
		print 'No line detected.'
	print 'Moment Zero G:',G_M0
	print 'Moment Zero M20:',M20_M0

	#if nradii==1:
	#	xt=[];xt.append(float(XLIST));XLIST=xt
	#	vct=[];vct.append(float(VLIST));VLIST=vct
	#	vdt=[];vdt.append(float(VDS));VDS=vdt

	if dobb:
		if M0emission:

			if fitz==-1:
				finalz=(linefreq/(L_head['restFRQ']/(1+unp.uarray(VSavg,VSstd)/(cspl))))-1
				realz=(linefreq/(L_head['restFRQ']/(1+VSavg/cspl)))-1
			else:
				finalz=fitz
				realz=fitz
			print "z: "+str(finalz)
			print 'vsys: ',unp.uarray(VSavg,VSstd)
			factor=CC.getcosmos(realz,cosmoparams[0],cosmoparams[1],cosmoparams[2])[3]
			print realz,factor
			FINAL_V='['
			print "(X,VROT,dVROT,VDISP,dVDISP)[kpc,km/s,km/s,km/s,km/s]: [",
			if nradii>1:
				for i in range(len(XLIST)):
					print '['+str(round(XLIST[i]*factor,2))+','+str(round(VLIST[i],2))+','+str(round(VRLIST[i],2))+','+str(round(VDS[i],2))+','+str(round(VDS_ERR[i],2))+']',
					FINAL_V+='['+str(round(XLIST[i]*factor,2))+','+str(round(VLIST[i],2))+','+str(round(VRLIST[i],2))+','+str(round(VDS[i],2))+','+str(round(VDS_ERR[i],2))+']'
					if i!=len(XLIST)-1:print ',',
			else:
					print XLIST,'Z',VLIST,'Z',VRLIST,'Z',VDS,'Z',VDS_ERR
					print '['+str(round(float(XLIST)*factor,2))+','+str(round(float(VLIST),2))+','+str(round(float(VRLIST),2))+','+str(round(float(VDS),2))+','+str(round(float(VDS_ERR),2))+']',
			FINAL_V+=']'
			print ']'
			print "INCL_MORPH: "+str(unp.arccos(M_fwhmy/M_fwhmx)*180/np.pi)+" deg"
			print "INCL_KINEM: "+str(round(INavg,2))+"+/-"+str(round(INstd,2))+" deg"
			print "PA_MORPH  : "+str(round(splituarray(M_theta)[0],2))+'+/-'+str(round(splituarray(M_theta)[1],2))+" deg"
			print "PA_KINEM  : "+str(round(PAavg,2))+"+/-"+str(round(PAstd,2))+" deg"
			#hdu_L = fits.open(cubename); L_head=fits.getheader(cubename,0); hdu_L.close()
			

			if nradii>1:
				rad_m=((3.086E+19)*factor*XLIST[len(XLIST)-1])
				erad_m=((3.086E+19)*factor*( XLIST[len(XLIST)-1]-XLIST[len(XLIST)-2] ))/2.
				v_ms=(1E+3)*VLIST[len(VLIST)-1]
				ev_ms=(1E+3)*VRLIST[len(VRLIST)-1]
				vd_ms=(1E+3)*VDS[len(VLIST)-1]
				evd_ms=(1E+3)*VDS_ERR[len(VLIST)-1]
			else:
				rad_m=(3.086E+19)*factor*float(XLIST)
				erad_m=(3.086E+19)*factor*(float(XLIST)/2.)
				v_ms=(1E+3)*float(VLIST)
				ev_ms=(1E+3)*float(VRLIST)
				vd_ms=(1E+3)*float(VDS)
				evd_ms=(1E+3)*float(VDS_ERR)
			gravc=6.67E-11
			rad_m_u=unp.uarray(rad_m,erad_m)
			v_ms_u=unp.uarray(v_ms,ev_ms)
			vd_ms_u=unp.uarray(vd_ms,evd_ms)
			mdyn=(((v_ms_u)**2)*rad_m_u/gravc)/(1.99E+30)
			mdyn2=3.4*(((vd_ms_u)**2)*rad_m_u/gravc)/(1.99E+30)

			print "log (Mdyn_rot/Msol): "+str(unp.log10(mdyn))
			print "log (Mdyn_dis/Msol): "+str(unp.log10(mdyn2))


			'''
			#Get residual:
			totres=0.;chi2=0.
			for tempspecindex in range(ZLEN):
				if bigspec_D[tempspecindex]!=0:
					chi2+=((bigspec_D[tempspecindex]-bigspec_M[tempspecindex])**2)/abs(bigspec_D[tempspecindex])
					totres+=abs(bigspec_D[tempspecindex]-bigspec_M[tempspecindex])
			#print 'For VDISP=',VDISP,'CHI^2=',chi2,'CHI^2/NCHAN=',chi2/ZLEN,'totres',totres,'totres/nchan',totres/ZLEN
			print 'For VDISP=',VDISP,'totres/nchan',totres/ZLEN
			'''


		#
		if MakeChanMaps:
			numr_CM=abs(m0_ch_high-m0_ch_low)+1
			numc_CM=2
			fig, axes = plt.subplots(numr_CM,numc_CM)
			fig.suptitle(sourcefile2,fontsize=fs2,weight='bold')

			#Data is already in temp_L
			#Model is already in temp_MOD

			for CM_kz in range(numr_CM):

				tempCH=m0_ch_low+CM_kz

				#Get data for chan of D
				CM_dat=np.zeros((temp_L.shape[0],temp_L.shape[1]))
				for CM_ix in range(temp_L.shape[0]):
					for CM_jy in range(temp_L.shape[1]):
						CM_dat[CM_ix,CM_jy]=temp_L[CM_ix,CM_jy,tempCH]

				#Get data for chan of M
				CM_mod=np.zeros((temp_MOD.shape[0],temp_MOD.shape[1]))
				for CM_ix in range(temp_MOD.shape[0]):
					for CM_jy in range(temp_MOD.shape[1]):
						CM_mod[CM_ix,CM_jy]=temp_MOD[CM_ix,CM_jy,tempCH]

				#Get data for chan of R
				CM_res=np.zeros((temp_MOD.shape[0],temp_MOD.shape[1]))
				for CM_ix in range(temp_MOD.shape[0]):
					for CM_jy in range(temp_MOD.shape[1]):
						CM_res[CM_ix,CM_jy]=CM_dat[CM_ix,CM_jy]-CM_mod[CM_ix,CM_jy]

				#Plot data
				axes[CM_kz,0].imshow(CM_dat)
				lev_both=[2,3,5]
				axes[CM_kz,0].contour(CM_mod,[X*chrms for X in lev_both], colors='k') 
				makepretty2(False,0,axes[CM_kz,0],xy0[0],xy0[1],False,False)
				axes[CM_kz,1].imshow(CM_res)
				makepretty2(False,0,axes[CM_kz,1],xy0[0],xy0[1],False,False)
				xpostext=axes[0,0].get_xlim()[0]+0.05*abs(axes[0,0].get_xlim()[0]-axes[0,0].get_xlim()[1])
				ypostext=axes[0,0].get_ylim()[1]-0.13*abs(axes[0,0].get_ylim()[0]-axes[0,0].get_ylim()[1])
				vch=int(fn.CHtoV(L_head,tempCH))
				axes[0,0].text(xpostext,ypostext,str(vch)+"km/s",bbox=dict(facecolor='white', alpha=1.0))
				plt.tight_layout()
				plt.savefig(sourcefold+sourcefile.replace('.','_')+'CHMAP.png',dpi=300,bbox_inches='tight')


		#W15 NUMBER 1
		#Does the VF show a continuous gradient?
		#Get M1 data
		ia=image()
		ia.open(D_M012[1])
		PA_temp=float(PAavg)*np.pi/180.
		length_temp=float(length)
		[x_temp,y_temp]=[M_RA_px,M_DEC_px]
		[x1,y1]=[x_temp-(length_temp/2.)*np.sin(PA_temp),y_temp+(length_temp/2.)*np.cos(PA_temp)]
		[x2,y2]=[x_temp+(length_temp/2.)*np.sin(PA_temp),y_temp-(length_temp/2.)*np.cos(PA_temp)]
		rec=ia.getslice(x=[x1,x2],y=[y1,y2])
		dist_temp=rec['distance']
		v_temp=rec['pixel']
		goodm1_D=[];goodm1_V=[]
		for i in range(len(dist_temp)):
			if abs(v_temp[i])>1E+1 and abs(v_temp[i])<1E+4:
				goodm1_D.append(dist_temp[i])
				goodm1_V.append(v_temp[i])
		flatguess_t=np.array([0.,np.mean(goodm1_V)])
		popt_t, pcov_t = curve_fit(polyn_t, np.array(goodm1_D), np.array(goodm1_V), p0=flatguess_t)
		plt.plot(goodm1_D,goodm1_V,linestyle='None',marker='o');plt.show()
		W15_1=abs(popt_t[0])>3*np.sqrt(pcov_t[0,0])

		#W15 NUMBER 2
		#Average VC(BB) > Average VD(BB), across all rings
		W15_2=None
		if M0emission:
			if not nradii==1:
				W15VC=0.;W15VD=0.
				for i in range(len(XLIST)):
					W15VC+=VLIST[i]/len(XLIST)
					W15VD+=VDS[i]/len(XLIST)
				W15_2=(W15VC>W15VD)
			else:
				W15_2=(VLIST>VDS)


		#W15 NUMBER 3
		#Average of two M1 extreme spaxels is close to M2 max
		W15_3=None
		if M0emission:
			#Get two extreme M1D locations
			M1ext=[1E+10,-1E+10,[0,0],[0,0]];
			for i in range(XLEN):
				for j in range(YLEN):
					if temp_M1[i,j]<M1ext[0]:
						M1ext[0]=temp_M1[i,j]
						M1ext[2]=[i,j]
					if temp_M1[i,j]>M1ext[0]:
						M1ext[1]=temp_M1[i,j]
						M1ext[3]=[i,j]
			avgm1loc=[0.5*(M1ext[2][0]+M1ext[3][0]),0.5*(M1ext[3][1]+M1ext[3][1])]
			#Get peak of M2D map
			M2ext=[-1E+10,[0,0]];
			for i in range(XLEN):
				for j in range(YLEN):
					if temp_M2[i,j]>M2ext[0]:
						M2ext[1]=[i,j]
						M2ext[0]=temp_M2[i,j]
			#Get difference
			W15_3_diff=np.sqrt((avgm1loc[0]-M2ext[1][0])**2+(avgm1loc[1]-M2ext[1][1])**2)
			W15_3=(W15_3_diff<3)
			print 'W15_3:',W15_3_diff


		#W15 NUMBER 4
		#PA difference<30deg
		W15_4=None
		if M0emission:
			temppa=0
			if PAavg>180.: temppa=180.
			W15_4=abs(M_PA-(PAavg-temppa))<30.
			#print M_PA,PAavg,temppa

		#W15 NUMBER 5
		#Average of two M1 extreme spaxels is close to M0 max
		W15_5=None
		if M0emission:
			#Get peak of M0D map
			M0ext=[-1E+10,[0,0]];
			for i in range(XLEN):
				for j in range(YLEN):
					if temp_M_D[i,j]>M0ext[0]:
						M0ext[1]=[i,j]
						M0ext[0]=temp_M_D[i,j]
			#Get difference
			W15_5_diff=np.sqrt((avgm1loc[0]-M0ext[1][0])**2+(avgm1loc[1]-M0ext[1][1])**2)
			W15_5=(W15_5_diff<3)
			print 'W15_5:',W15_5_diff

		print W15_1,W15_2,W15_3,W15_4,W15_5

	if not dobb:
		gm20_file=open('/Users/garethjones/Desktop/GM20_JAN2021.txt','a')
		gm20_file.write(sourcefile.replace('.SB2','')+' '+str(G_M0)+' '+str(M20_M0)+'\n')
		gm20_file.close()
	else:
		gm20_file=open('/Users/garethjones/Desktop/GM20_May2021_V0.txt','a')
		#str(round(splituarray(M_theta)[0],2))+'+/-'+str(round(splituarray(M_theta)[1],2))
		FINAL_PA_M=str(int(round(splituarray(M_theta)[0],0)))+'+/-'+str(int(round(splituarray(M_theta)[1],0)))
		FINAL_PA_K=str(int(round(PAavg,0)))+"+/-"+str(int(round(PAstd,0)))
		try:
			FINAL_I_M_temp=unp.arccos(M_fwhmy/M_fwhmx)*180/np.pi 
			FINAL_I_M=str(int(round(splituarray(FINAL_I_M_temp)[0],0)))+'+/-'+str(int(round(splituarray(FINAL_I_M_temp)[1],0)))
		except ValueError:
			FINAL_I_M='999+/-999'
		FINAL_I_K=str(int(round(INavg,0)))+"+/-"+str(int(round(INstd,0)))
		FINAL_Z=str(splituarray(finalz)[0])+'+/-'+str(splituarray(finalz)[1])
		FINAL_MDYN=str(unp.log10(mdyn))
		#gm20_file.write(sourcefile.replace('.SB2','')+" "+str(G_M0)+" "+str(G_CO)+" "+str(M20_M0)+" "+str(M20_CO)+'\n')
		#gm20_file.write(sourcefile.replace('.SB2','')+' '+FINAL_PA_K+' '+FINAL_PA_M+' '+FINAL_I_K+' '+FINAL_I_M+" "+str(G_M0)+' '+str(M20_M0)+' '+str(W15_1)+' '+str(W15_2)+' '+str(W15_3)+' '+str(W15_4)+' '+str(W15_5)+' '+FINAL_V+' '+str(realz)+''+'\n')
		gm20_file.write(sourcefile2+' '+FINAL_PA_K+' '+FINAL_PA_M+' '+FINAL_I_K+' '+FINAL_I_M+" "+str(G_M0)+' '+str(M20_M0)+' '+str(W15_1)+' '+str(W15_2)+' '+str(W15_3)+' '+str(W15_4)+' '+str(W15_5)+' '+FINAL_V+' '+FINAL_Z+' '+str(FINAL_MDYN)+'\n')
		gm20_file.close()


	#Clean up.
	badfiles=[outputfolder+sourcefile+'NONE_mom0th.fits',outputfolder+sourcefile+'NONE_mom1st.fits',outputfolder+sourcefile+'NONE_mom2nd.fits',sourcefold+'NONE_mom0th.fits',sourcefold+'NONE_mom1st.fits',sourcefold+'NONE_mom2nd.fits']
	badfolds=[outputfolder+sourcefile+'_MODELm0.im',outputfolder+sourcefile+'_MODELm1.im',outputfolder+sourcefile+'_MODELm2.im',outputfolder+'maps',outputfolder+'pvs',sourcefold+sourcefile+'_m2_script_SIG_RED.im',sourcefold+sourcefile+'_m0_script_FUL_NOI.im',sourcefold+sourcefile+'_m0_script_FUL_SIG_RED.im',sourcefold+sourcefile+'_m1_script_SIG_RED.im',sourcefold+sourcefile+'_m0_script_FUL.im']
	for badfile in badfiles:
		try:
			os.remove(badfile)
			print "Deleted: ",badfile
		except OSError:
			pass
	for badfolder in badfolds:
		try:
			shutil.rmtree(badfolder)
			print "Deleted: ",badfolder
		except OSError:
			pass

	print 'Used ch',m0_ch_low,'to',m0_ch_high,', or nu=',VtoNU(fn.CHtoV(L_head,m0_ch_low),L_head['restFrq']),'to',VtoNU(fn.CHtoV(L_head,m0_ch_high),L_head['restFrq']),'GHz'
	print sourcefile2+' & $'+str(FINAL_PA_M).replace('+/-','\pm')+'$ & $'+str(FINAL_PA_K).replace('+/-','\pm')+'$ & $'+str(FINAL_I_M).replace('+/-','\pm')+'$ & $'+str(FINAL_I_K).replace('+/-','\pm')+'$ & $'+str(FINAL_MDYN).replace('+/-','\pm')+' & '+str(W15_1)+' & '+str(W15_2)+' & '+str(W15_3)+' & '+str(W15_4)+' & '+str(W15_5)
