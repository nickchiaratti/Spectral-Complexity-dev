![background image](images/HLS_User_Guide_V2001.png)

**Harmonized Landsat And Sentinel-2**

**(HLS) Product User Guide**

*Product Version 2.0*

Junchang Ju, Christopher Neigh, Madhu Sridhar, Martin Claverie, Sergii   
Skakun, Jean-Claude Roger, Eric Vermote, Jennifer Dungan, Lavanya   
Ashokkumar

Principal Investigator:

Christopher Neigh

, NASA/GSFC

Correspondence email address:

[christopher.s.neigh@nasa.gov](mailto:christopher.s.neigh@nasa.gov)

<br />

<br />

<br />

<br />

Last updated: April, 2026

<br />

![background image](images/HLS_User_Guide_V2002.png)

Table of Contents

[Acronyms](HLS_User_Guide_V2.html#3)

[](HLS_User_Guide_V2.html#3)

[1. Introduction](HLS_User_Guide_V2.html#6)

[](HLS_User_Guide_V2.html#6)

[2. New in V2.0](HLS_User_Guide_V2.html#6)

[3. Products Overview](HLS_User_Guide_V2.html#7)

[3.1. Input Spectral Data](HLS_User_Guide_V2.html#7)

[3.2. Overall HLS Processing Flowchart](HLS_User_Guide_V2.html#9)

[](HLS_User_Guide_V2.html#9)

[3.3. Products specifications](HLS_User_Guide_V2.html#9)

[](HLS_User_Guide_V2.html#9)

[3.4. Spectral Bands](HLS_User_Guide_V2.html#10)

[](HLS_User_Guide_V2.html#10)

[3.5. Output Projection and Gridding](HLS_User_Guide_V2.html#11)

[](HLS_User_Guide_V2.html#11)

[4. Description of Algorithms](HLS_User_Guide_V2.html#12)

[](HLS_User_Guide_V2.html#12)

[4.1. Atmospheric Correction](HLS_User_Guide_V2.html#12)

[](HLS_User_Guide_V2.html#12)

[4.2. Cloud Masking And The Quality Assessment Band](HLS_User_Guide_V2.html#12)

[4.3. Spatial Co-Registration Of Landsat And Sentinel-2 Data](HLS_User_Guide_V2.html#13)

[4.4. View And Illumination Angles Normalization](HLS_User_Guide_V2.html#13)

[4.5. Bandpass Adjustment](HLS_User_Guide_V2.html#16)

[](HLS_User_Guide_V2.html#16)

[5. Spatial and Temporal Coverage](HLS_User_Guide_V2.html#17)

[6. Product Formats](HLS_User_Guide_V2.html#18)

[6.1. File Format](HLS_User_Guide_V2.html#18)

[](HLS_User_Guide_V2.html#18)

[6.2. L30 and S30 Products](HLS_User_Guide_V2.html#20)

[](HLS_User_Guide_V2.html#20)

[6.3. The Sun And View Angles](HLS_User_Guide_V2.html#21)

[6.4. Quality Assessment Layer](HLS_User_Guide_V2.html#22)

[6.5. Metadata](HLS_User_Guide_V2.html#23)

[7. Known Issues](HLS_User_Guide_V2.html#25)

[References](HLS_User_Guide_V2.html#25)

[](HLS_User_Guide_V2.html#25)

[Appendix A: How To Decode The Bit-Packed QA](HLS_User_Guide_V2.html#30)

![background image](images/HLS_User_Guide_V2003.png)

Acronyms

**Missions, Programs, and Organization**

NASA

National Aeronautics and Space Administration

USGS

United States Geological Survey

ESA

European Space Agency

CEOS

Committee on Earth Observation Satellites

GSFC

Goddard Space Flight Center (NASA)

EROS

Earth Resources Observation and Science (USGS) Center

NOAA

National Oceanic and Atmospheric Administration

MODIS

Moderate Resolution Imaging Spectroradiometer

VIIRS

Visible Infrared Imaging Radiometer Suite

CEOS ACIX CEOS Atmospheric Correction Intercomparison Exercise   
HLS

Harmonized Landsat and Sentinel-2

CMG

Climate Modeling Grid

**Satellite Sensors and Instruments**

OLI

Operational Land Imager (on Landsat 8/9)

TIRS

Thermal Infrared Sensor (on Landsat 8/9)

MSI

MultiSpectral Instrument (on Sentinel-2A/B/C)

TM

Thematic Mapper

ETM+

Enhanced Thematic Mapper Plus (on Landsat 7)

S2A, S2B,   
S2C

Sentinel-2A, Sentinel-2B, Sentinel-2C satellites

L8, L9

Landsat 8 and Landsat 9 satellites

**Data Products and Processing Levels**

L30

Harmonized Landsat 8/9 30 m Surface Reflectance (SR) and NBAR   
Product

S30

Harmonized Sentinel-2 30 m Surface Reflectance (SR) and NBAR   
Product

L1TP

Level-1 Terrain Precision-corrected Landsat Product

L1C

Sentinel-2 Level-1C Top-of-Atmosphere Reflectance Product   
Collection 2 (USGS Landsat Data Reprocessing Collection)

RT

Real-Time (Landsat data stream used for low-latency processing)

SR

Surface Reflectance  
![background image](images/HLS_User_Guide_V2004.png)

NBAR

Nadir BRDF-Adjusted Reflectance

QA

Quality Assessment (per-pixel quality mask)

COG

Cloud-Optimized GeoTIFF

HDF

Hierarchical Data Format (previous file format for HLS V1.4)

TOA

Top-of-Atmosphere (reflectance or brightness temperature)

BT

Top-of-Atmosphere Brightness temperature

GDAL

Geospatial Data Abstraction Library

KML

Keyhole Markup Language

**Algorithms and Models**

LaSRC

Land Surface Reflectance Code (NASA/USGS atmospheric   
correction algorithm)

6SV

Second Simulation of the Satellite Signal in the Solar Spectrum,   
Vector version (radiative transfer model)

Fmask

Function of Mask (cloud, shadow, snow, water detection algorithm)

AROP

Automated Registration and Orthorectification Package

BRDF

Bidirectional Reflectance Distribution Function

AOT

Aerosol Optical Thickness

RSR

Relative Spectral Response

GRI

Global Reference Image

c-factor

Coefficient used for BRDF normalization (Roy et al., 2016)

**Coordinate Systems and Gridding**

UTM

Universal Transverse Mercator projection

MGRS

Military Grid Reference System (Sentinel-2 tiling system)

WRS

Worldwide Reference System (Landsat path-row grid)

CE90

Circular Error at 90% confidence (geolocation accuracy metric)

**Ancillary and Validation Datasets**

MCD43

MODIS BRDF/Albedo Product

ACIX-I /   
ACIX-II

Atmospheric Correction Intercomparison Exercise Phases I \& II

GSHHG

Global Self-consistent, Hierarchical, High-resolution Geography   
(NOAA shoreline dataset)

MOD09CMA MODIS Daily CMG Atmospheric Parameters (water vapor and

ozone inputs)

**Data Layers , Metadata and Supporting References**

SZA

Sun Zenith Angle  
![background image](images/HLS_User_Guide_V2005.png)

SAA

Sun Azimuth Angle

VZA

View Zenith Angle

VAA

View Azimuth Angle

B##

Spectral Band Number (e.g., B04 = Red Band)

ACCODE

Atmospheric Correction Code version (LaSRC version used)

FMASK

File name for the per-pixel Fmask QA layer

ATBD

Algorithm Theoretical Basis Document

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

<br />

![background image](images/HLS_User_Guide_V2006.png)

1. Introduction

The Harmonized Landsat and Sentinel-2 (HLS) project is a NASA initiative and a   
collaboration with the United States Geological Survey (USGS) to produce compatible   
surface reflectance (SR) data from a virtual constellation of satellite sensors: the Operational   
Land Imager (OLI) and Multi-Spectral Instrument (MSI) onboard the US Landsat 8/9 and the   
ESA Sentinel-2 A/B/C (S2A, S2B, S2C) remote sensing satellites, respectively. The HLS   
project derives seamless products from OLI and MSI using a set of algorithms for   
atmospheric correction, cloud and cloud-shadow masking, spatial co-registration and   
common gridding, view angle normalization and spectral bandpass adjustment. The   
combined measurement enables global land observation every 1.6 days on average at   
moderate (30 m) spatial resolution (Zhou et al. 2025; Ju et al. 2025). The HLS data products   
can be regarded as the building blocks for a "data cube" so that a user may examine any pixel   
through time and treat the near-daily reflectance time series as though it came from a single   
sensor (Masek et al., 2006).

The HLS SR suite contains two products, L30 and S30, the 30-m spatial resolution   
nadir-view surface reflectance, derived from Landsat L1TP (Collection 2, Crawford et al.   
2023) and Sentinel-2 L1C input (Drusch et al. 2012), and gridded into the same MGRS tiles   
with a 30 m pixel size. The temporal coverage of the HLS dataset extends from the first   
Landsat 8 acquisitions in 2013 and the first Sentinel-2 acquisitions in 2015 to the present.

2. New in V2.0

HLS V2.0 builds on V1.4 by updating and improving processing algorithms, expanding   
spatial coverage, and providing validation.

The key improvements in the HLS V2.0 include

the following:

●

*Global coverage*

. All of the global land area is covered, including major islands but

excluding Antarctica.

●

*Input data*

. The new Landsat Collection-2 (C2) data from USGS are used as input;

better geolocation is expected as C2 data are produced from the refined ground   
control points (GCP) with the Sentinel-2 Global Reference Image (GRI) as an   
absolute reference.

●

*Atmospheric correction*

. A USGS C version of LaSRCv3.5.5 is applied for both

Landsat and Sentinel-2 data for computational speedup and bug fixes. LaSRCv3.5.5   
has been validated for both Landsat 8 and Sentinel-2 within the CEOS ACIX-I   
(Atmospheric

Correction

Inter-Comparison

eXercise,

<http://calvalportal.ceos.org/projects/acix>

[).](http://calvalportal.ceos.org/projects/acix)  
![background image](images/HLS_User_Guide_V2007.png)

●

*QA band*

. The cloud and shadow bits of the QA band are exclusively based on

Fmask

1

, consistently for the two HLS products (S30 and L30).

●

*BRDF adjustment*

. BRDF adjustment mainly normalizes the view angle effects, with

the solar zenith angle largely intact. This adjustment is applied to the Sentinel-2   
red-edge bands as well.

●

*Sun and view angle bands.*

These bands are provided.

●

*Product format*

. The surface reflectance and QA products are delivered in individual

Cloud Optimized GeoTIFF (COG) files to allow for spectral and spatial subsetting in   
applications. Previously, HDF file format was used in the HLS V1.4 and before.   

3. Products Overview

3.1. Input Spectral Data

The Operational Land Imager (OLI) sensor is a medium spatial resolution multispectral   
imager onboard the Landsat 8/9 satellites, in a sun-synchronous orbit with a 705 km altitude   
and a 16 day repeat cycle for a single satellite, and 8 days combined. The sensor acquires   
data with a 15° field of view resulting in approximately a 185 km image swath. The OLI   
sensor has nine solar reflective bands, and the reflective data are co-registered with the data   
from the 2-band Thermal Infrared Sensor (TIRS) onboard the same observatories (Irons et   
al., 2012). The native spatial resolution is 30 m for OLI and 100 m for TIRS, but TIRS data   
are resampled to 30 m for distribution. The HLS V2.0 processing only uses L1TP images   
with the best geolocation quality in the Collection 2 Landsat 8 and 9 archive. Within L1TP,   
images are further separated into Tier 1 (T1) and Tier 2 (T2) based on geolocation quality,   
with the T2 image geolocation error up to 30 m in the

*x*

or

*y*

direction (Storey et al., 2014).

Ideally, only T1 images should be used for HLS, but L1TP T2 imagery makes up only \~5%   
of the whole L1TP category (see Ju et al. 2025 for further details).   

The Sentinel-2 Multi-Spectral Instrument (MSI) is onboard the Sentinel-2A, -2B, and -2C   
satellites in a sun-synchronous orbit with a 786 km altitude and a 10 day repeat cycle for a   
single satellite. The sensor has a 20.6° field of view corresponding to an image swath of   
approximately 290 km. The spatial resolution varies with the spectral bands: 10 m for the   
visible bands and the broad NIR band; 20 m for the red edge, the narrow NIR and the SWIR   
bands; and 60 m for the atmospheric bands (Drusch et al., 2012). Before the global   
application of the geolocation Global Reference Image (GRI) in August 2021, the Sentinel-2   
L1C data have a geolocation uncertainty of 12 m circular error at 90% probability (CE90),   
but the subsequent use of GRI reduced the uncertainty to 5.1 m (CE90) (Rengarajan et al,   
2024., Yan et al., 2018). The HLS V2.0 production was started in September 2021, before   
ESA started to reprocess historical data with the GRI deployment. As a result, the HLS

1

https://github.com/GERSL/Fmask  
![background image](images/HLS_User_Guide_V2008.png)

Sentinel-2 product (S30) from images observed before August 2021 have poorer geolocation   
quality but are acceptable for most applications because the geolocation error is less than half   
of a 30 m HLS pixel (Storey et al., 2016).

HLS V2.0 uses the Level-1 (top of atmosphere) data as input.

[Table 1]()

provides an overview

of Landsat 8/9 and Sentinel-2A/B/C data characteristics.

*Table 1. Input satellite sensor characteristics: Landsat 8/9 OLI--TIRS and
Sentinel-2A/B/C MSI. References:*

[*https://science.nasa.gov/mission/landsat/spectral-bands-and-applications/*](https://science.nasa.gov/mission/landsat/spectral-bands-and-applications/)

[*https://sentinels.copernicus.eu/-/copernicus-sentinel-2c-spectral-response-functions*](https://sentinels.copernicus.eu/-/copernicus-sentinel-2c-spectral-response-functions)

**Landsat 8/
OLI-TIRS**

**Landsat 9/
OLI-TIRS**

**Sentinel-2A/
MSI**

**Sentinel-2B/
MSI**

**Sentinel-2C/
MSI**

Launch date

February 11, 2013 September 27, 2021

June 23, 2015 March 7, 2017 September 4, 2024

Equatorial crossing   
time

10:00 a.m.

10:30 a.m.

Spatial resolution 30 m (OLI) / 100 m (TIRS)

10 m / 20 m / 60 m (given below in parentheses for   
each spectral band)

Swath/   
Field of view

185 km / 15°

290 km / 20.6°

Spectral bands:   

Coastal/Aerosol   
Blue   
Green   
Red

wavelength in

[](https://www.codecogs.com/eqnedit.php?latex=%5Cmu%20m#0)and spatial

resolution in parentheses   

0.43-0.45 (30m)   
0.45-0.51 (30m)   
0.53-0.59 (30m)   
0.64-0.67 (30m)

<br />

0.433-0.453   
0.460-0.524   
0.543-0.578   
0.650-0.680

<br />

0.433-0.453   
0.459-0.524   
0.542-0.577   
0.650-0.681

<br />

0.434-0.455 (60m)   
0.457-0.522 (10m)   
0.543-0.579 (10m)   
0.652-0.682 (10m)

Red edge

Not Applicable

0.698-0.712   
0.734-0.748   
0.774-0.794

0.697-0.712   
0.733-0.747   
0.771-0.791

0.700-0.715 (20m)   
0.734-0.749 (20m)   
0.775-0.796 (20m)

NIR (broad)

Not Applicable

0.783-0.901

0.783-0.898

0.787-0.901 (10m)

NIR (narrow)

0.85-0.88 (30m)

0.855-0.875

0.855-0.875

0.856-0.876 (20m)

Water Vapor

Not Applicable

0.936-0.956

0.934-0.954

0.938-0.958 (60m)

Cirrus

1.36-1.38 (30m)

1.359-1.389

1.363-1.393

1.356-1.389 (60m)

SWIR 1   
SWIR 2

1.57-1.65 (30m)   
2.11-2.29 (30m)

1.570-1.658   
2.108-2.287

1.565-1.658   
2.094-2.275

1.567-1.656 (20m)   
2.102-2.284 (20m)

Thermal Infrared 1   
Thermal Infrared 2

10.60-11.19 (100m)   
11.50-12.51 (100m)

Not Applicable

![background image](images/HLS_User_Guide_V2009.png)

3.2. Overall HLS Processing Flowchart

A group of processing methods are applied to generate L30 and S30 [(](HLS_User_Guide_V2.html#9)

[Figure. 1)](HLS_User_Guide_V2.html#9)

with some

methods common to both products and some unique to L30 or S30. LaSRC (Vermote et al.   
2018) is used for atmospheric correction and Fmask (Qiu et al. 2019) for cloud masking   
(QA). Landsat data are gridded into the tiles that MSI uses; Landsat and Sentinel-2 pixels are   
resampled to 30 m. Surface reflectance is corrected for the view angle effects (Roy et al.   
2016), and MSI bandpasses are adjusted to match Landsat. A detailed description of   
processing methods can be found in Section 4.

*Figure 1. HLS processing flow.*

3.3. Products specifications

The HLS product characteristics are given in

[Table 2](HLS_User_Guide_V2.html#9)

.

*Table 2. HLS products specifications.*

**Product Name**

**L30**

**S30**

Input sensor

Landsat 8/9 OLI+TIRS

Sentinel-2 A/B/C MSI

Input data level

USGS Collection-2 L1TP   
(top-of-atmosphere)

ESA

Sentinel-2

L1C

(top-of-atmosphere)  
![background image](images/HLS_User_Guide_V2010.png)

Spatial resolution

30 m (all bands resampled)

30 m (resampled from 10/20/60   
m)

Spectral bands

11 bands (OLI + TIRS)

13 bands (MSI)

BRDF-adjusted

Yes (except band 09)

Yes (except bands 09, 10)

Bandpass-adjusted

No (HLS uses OLI bandpass   
as reference)

Adjusted to OLI-like (except red   
edge, water vapor and cirrus   
bands)

Projection

UTM

UTM

Tiling system

MGRS (110×110 km)

MGRS (110×110 km)

File format

Cloud Optimized GeoTIFF (COG)

3.4. Spectral Bands

All Landsat 8/9 OLI and Sentinel-2 MSI reflective spectral band nomenclatures are retained   
in the HLS products (

[Table 3](HLS_User_Guide_V2.html#10)

).

*Table 3. HLS V2.0 spectral bands nomenclature. The spectral band wavelengths are nominal
values as they can vary slightly with the satellites. The character - indicates Not Applicable.*

**Band name**

**OLI band**

**number**

**MSI band**

**number**

**HLS band**

**code name**

**L8/L9**

**HLS band**

**code name**

**S2-A/B/C**

**Wavelength**

**(μm)**

Coastal

Aerosol

1

1

B01

B01

0.43 -- 0.45

\*

Blue

2

2

B02

B02

0.45 -- 0.51

\*

Green

3

3

B03

B03

0.53 -- 0.59

\*

Red

4

4

B04

B04

0.64 -- 0.67

\*

Red-Edge 1

-

5

-

B05

0.69 -- 0.71

\*\*

Red-Edge 2

-

6

-

B06

0.73 -- 0.75

\*\*

Red Edge 3

-

7

-

B07

0.77 -- 0.79

\*\*

NIR Broad

-

8

-

B08

0.78 --0.88

\*\*  
![background image](images/HLS_User_Guide_V2011.png)

NIR Narrow

5

8A

B05

B8A

0.85 -- 0.88

\*

SWIR 1

6

11

B06

B11

1.57 -- 1.65

\*

SWIR 2

7

12

B07

B12

2.11 -- 2.29

\*

Water vapor

-

9

-

B09

0.93 -- 0.95

\*\*

Cirrus

9

10

B09

B10

1.36 -- 1.38

\*

Thermal

Infrared 1

10

-

B10

-

10.60 --11.19

\*

Thermal

Infrared 2

11

-

B11

-

11.50 --12.51

\*

\*

from OLI specifications

\*\*

from MSI specifications

3.5. Output Projection and Gridding

HLS has adopted the tiling system used by ESA for Sentinel-2. The tiles are in the Universal   
Transverse Mercator (UTM) projection and are 109,800 m (110 km nominally) on one side   
(ESA, 2018). The tiling system is aligned with the Military Grid Reference System (MGRS).   
The UTM system divides the Earth's surface into 60 longitude zones, each 6° of longitude in   
width, numbered 1 to 60 from 180° West to 180° East. Each UTM zone is divided into   
latitude bands of 8°, labeled with letters C to X from South to North (excluding I and O). A   
useful mnemonic is that latitude bands N and later are in the Northern Hemisphere. Each 6° ×   
8° polygon (grid zone) is further divided into the 110 km × 110 km Sentinel-2 tiles labeled   
with letters. For example, tile 11SPC is in UTM zone 11, latitude band S (in the Northern   
Hemisphere), and labeled P in the east-west direction and C in the south-north direction   
within grid zone 11S. Users should note that there is horizontal and vertical overlap of   
around 8-10 km between two adjacent tiles in the same UTM zone. For the two adjacent tiles   
from two neighboring UTM zones, the overlap may be much greater. A KML file produced   
by ESA showing the location of all Sentinel-2 tiles is available at:

[https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__2
0151209T095117_V20150622T000000_21000101T000000_B00.zip](https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip)

[](https://sentiwiki.copernicus.eu/__attachments/1692737/S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.zip)

One trivial difference from the ESA Sentinel-2 gridding is that HLS inherits the USGS   
Landsat UTM convention of keeping the

*y*

coordinate for the Southern Hemisphere negative,

therefore with no need for hemisphere specification. In contrast, some data providers adopt a   
convention of adding 10,000,000 meters to make the southern coordinate positive (i.e., use of   
a false northing 10,000,000) and thus need to indicate which hemisphere to avoid confusion.   
The end users need to be aware of this false northing difference for geospatial visualization  
![background image](images/HLS_User_Guide_V2012.png)

or mosaicking HLS products. However, most GIS tools (e.g., ArcGIS, QGIS, GDAL) will   
automatically detect the USGS convention.   

4. Description of Algorithms

The following sections provide a brief description of algorithms for the major steps in HLS   
data generation. Detailed descriptions of the algorithms can be found in Ju et al. (2025).

4.1. Atmospheric Correction

The surface reflectance derivation from Landsat and Sentinel-2 images uses Land Surface   
Reflectance Code (LaSRC), an atmospheric correction based on the 6SV radiative transfer   
algorithm (Vermote and Kotchenova, 2008), originally developed by Eric Vermote   
(NASA/GSFC) (Vermote et al., 2016, 2018) and adapted by USGS EROS for operational   
use. A detailed description of the operational Landsat surface reflectance and its ancillary   
layers are provided by the U.S. Geological Survey, 2024 (U.S. Geological Survey, 2024).   

LaSRC derives surface reflectance with the aerosol optical thickness retrieved from the   
images assuming a continental aerosol model and using ancillary data from a few sources.   
The ancillary atmospheric ozone and water vapor data for gaseous absorption correction are   
from the MODIS 0.05° CMG data (before May 2024) or from VIIRS 0.05° CMG (after May   
2024). Standard sea surface atmospheric pressure is adjusted for 0.05° resolution local   
elevation in molecular (Rayleigh) scattering correction. The aerosol optical thickness   
retrieval uses the spatially explicit ratios of the red to blue surface reflectances, also in the   
0.05° CMG grid, that are characterized from 10 year MODIS observations; the spectral ratio   
data are adjusted by the observed finer-resolution Landsat and Sentinel-2 spectral data to be   
processed (Vermote et al., 2016., 2018). HLS also includes the two thermal infrared bands   
from the Landsat 8/9 TIRS sensor in the L30 product -- these values are not atmospherically   
corrected, but are rescaled to apparent brightness temperature.

4.2. Cloud Masking And The Quality Assessment Band

HLS provides per-pixel masking of cloud, cloud shadow, snow, water, and aerosol optical   
thickness levels. In earlier versions of HLS, the cloud and shadow masks were a union of   
results from multiple sources, but in HLS V2.0, they were generated exclusively by Fmask   
4.7 (Zhu and Woodcock., 2012., Zhu et al., 2016), an update of the previous Fmask version   
reported in Qiu et al. (2017) and Qiu et al. (2019). Fmask was found to be one of the better   
performing algorithms when compared and assessed on a variety of test data sites (Skakun et   
al., 2017, 2022). Clouds and cloud shadow are dilated by 150 m during HLS processing, and   
the dilated area is labeled as "adjacent to clouds." The results from the Fmask classes (cloud,   
shadow, snow/ice, water, adjacency) are stored in bits 1-5 in the QA band. Fmask uses the  
![background image](images/HLS_User_Guide_V2013.png)

cirrus band in cloud detection, but a separate cirrus class is not created; cirrus is aggregated   
into the generic "cloud" class. The aerosol optical thickness level created during atmospheric   
correction is also incorporated into the per-pixel quality assessment mask (bits 6 and 7), like   
in HLS V1.4.   

4.3. Spatial Co-Registration Of Landsat And Sentinel-2 Data

HLS adopts the UTM-based MGRS tiling system in which ESA grids the Sentinel-2 L1C   
data. Although the USGS Landsat data are also in UTM projection and the HLS output pixel   
size is 30 m, resampling is needed in gridding the 30 m Landsat data into MGRS. This is   
because for USGS Landsat, the origin of the UTM coordinate system corresponds to the   
center of a 30 m pixel, but in the MGRS system the origin corresponds to a pixel corner. The   
HLS resampling of Landsat spectral data uses the cubic convolution method. If the input   
Landsat data are from a UTM zone adjacent to the MGRS one in consideration, reprojection   
is needed before resampling. The resampling of the accompanying categorical QA values   
only considers the inner 2 × 2 pixels in the cubic convolution kernel, and the presence of any   
label there will turn on the corresponding output QA bit. Because the 2 × 2 pixels may have   
different Fmask labels, an output HLS pixel may have multiple bits set to 1 in its QA byte;   
for the aerosol optical thickness level, the label of the highest aerosol level in the inner four   
pixels is chosen for output.   

The 10 m, 20 m and 60 m Sentinel-2 spectral data are resampled to 30 m with an   
area-weighted average. The categorical LaSRC aerosol thickness levels derived at 10 m and   
the Fmask labels derived at 20 m are resampled to 30 m from the overlapping input pixels   
with a rule similar to the Landsat categorical data resampling.

4.4. View And Illumination Angles Normalization

The L30 and S30 are Nadir BRDF-Adjusted Reflectance (NBAR), surface reflectance   
normalized for a nadir view direction and a slightly adjusted solar zenith angle, using the   
c-factor technique by Roy et al. (2016, 2017). The BRDF effects in Landsat and Sentinel-2   
data are most noticeable in the contrasting backward and forward view image pairs acquired   
a few days apart. During normalization, the view zenith angle is set to for all pixels, but   
the solar zenith angle is adjusted only by a trivial amount. For an image acquired on a given   
tile and a given day, the solar zenith angles at the tile center's latitude for the Landsat   
overpass time and for the Sentinel-2 overpass time are calculated for the day respectively,   
and the mean of the two solar zenith angle values is used for all the pixels in the tile on that   
day for NBAR derivation. This solar zenith angle setting for NBAR contrasts with the use of   
a latitude-dependent but temporally-constant solar zenith angle in HLS V1.4.

![background image](images/HLS_User_Guide_V2014.png)

The HLS BRDF normalization uses a set of constant BRDF coefficients derived from the 12   
month MODIS 500 m global BRDF product (MCD43A1 Version 6) for more than 15 billion   
pixels. The derived BRDF coefficients are applied to OLI and MSI bands equivalent to   
MODIS bands. For the normalization of the Sentinel-2 MSI red-edge bands (705, 740, and   
783 nm) that have no MODIS equivalents, the linearly interpolated BRDF coefficients from   
the enclosing MODIS red and NIR wavelength bands are used (Roy et al 2017). The   
technique has been evaluated using off-nadir (i.e., in the overlap areas of adjacent swaths)   
ETM+ data (Roy et al. 2016) and MSI data (Roy et al. 2017). BRDF coefficients for the three   
kernels (isotropic, geometric, and volumetric) of the Ross--Li semiempirical model are shown   
in the

[Table 4](HLS_User_Guide_V2.html#14)

[.](HLS_User_Guide_V2.html#14) The kernel definitions are described in the ATBD of the MOD43A1 product

(Strahler et al., 1999), and the operational implementation for MODIS BRDF/albedo product   
are described in Schaaf et al., 2002.   

*Table 4: BRDF coefficients used in the HLS V2.0 c-factor normalization (Roy et al., 2016,
2017; Ju et al., 2025).*

**Band name**

**HLS**

**band**

**code**

**name L8**

**HLS**

**band**

**code**

**name S2**

**Equivalent**

**MODIS**

**band**

**f**

**iso**

**f**

**geo**

**f**

**vol**

Coastal/Aerosol

B01

B01

3

0.0774

0.0079

0.0372

Blue

B02

B02

3

0.0774

0.0079

0.0372

Green

B03

B03

4

0.1306

0.0178

0.0580

Red

B04

B04

1

0.1690

0.0227

0.0574

Red-Edge 1

-

B05

-

0.2085

0.0256

0.0845

Red-Edge 2

-

B06

-

0.2316

0.0273

0.1003

Red-Edge 3

-

B07

-

0.2599

0.0294

0.1197

NIR Broad

-

B08

2

0.3093

0.0330

0.1535

NIR Narrow

B05

B8A

2

0.3093

0.0330

0.1535

SWIR 1

B06

B11

6

0.3430

0.0453

0.1154

SWIR 2

B07

B12

7

0.2658

0.0387

0.0639

The L30 and S30 NBAR are derived as:  
![background image](images/HLS_User_Guide_V2015.png)

[](https://www.codecogs.com/eqnedit.php?latex=%5Crho(%5Clambda%2C%5CTheta%5E%7Bnorm%7D)%20%3D%20c(%5Clambda%2C%5CTheta%5E%7Bnorm%7D%2C%5CTheta%5E%7Bsensor%7D)%20%5Ccdot%20%5Crho(%5Clambda%2C%5CTheta%5E%7Bsensor%7D)#0)

(1)

[](https://www.codecogs.com/eqnedit.php?latex=c(%5Clambda%2C%5CTheta%5E%7Bnorm%7D%2C%5CTheta%5E%7Bsenor%7D%20)%3D%5Cfrac%7Bf_%7Biso%7D%20(%5Clambda)%2Bf_%7Bgeo%7D(%5Clambda)%5Ccdot%20K_%7Bgeo%7D%20(%5CTheta%5E%7Bnorm%20%7D)%2Bf_%7Bvol%7D%20(%5Clambda)%5Ccdot%20K_%7Bvol%7D%20(%5CTheta%5E%7Bnorm%7D%20)%7D%7Bf_%7Biso%7D%20(%5Clambda)%2Bf_%7Bgeo%7D%20(%5Clambda)%5Ccdot%20K_%7Bgeo%7D%20(%5CTheta%5E%7Bsensor%7D)%2Bf_%7Bvol%7D%20(%5Clambda)%5Ccdot%20K_%7Bvol%7D%20(%5CTheta%5E%7Bsensor%7D%20)%20%7D#0)

(2)

where

[](https://www.codecogs.com/eqnedit.php?latex=%5Crho(%5Clambda%2C%5CTheta%5E%7Bsensor%7D)#0)and

are the LaSRC-derived surface reflectance and the

BRDF-normalized surface reflectance, respectively;

denotes the angle set of the

observed solar zenith, view zenith, and the relative azimuth; and[](https://www.codecogs.com/eqnedit.php?latex=%5CTheta%5E%7Bnorm%7D#0)

[](https://www.codecogs.com/eqnedit.php?latex=%5CTheta%5E%7Bnorm%7D#0)denotes the angle set

of the prescribed solar zenith, a [](https://www.codecogs.com/eqnedit.php?latex=0%5E%5Ccirc#0)view zenith, and accordingly a [](https://www.codecogs.com/eqnedit.php?latex=0%5E%5Ccirc#0)relative azimuth used in   
normalization. The c-factor is a ratio of the modeled surface reflectance for the normalization   
angle set to the modeled surface reflectance for the observation angle set.   

The slight adjustment to the solar zenith angle is made due to two considerations. First,   
Landsat and Sentinel-2 overpass the same latitude 30 minutes apart; on those rare days when   
Landsat and Sentinel-2 overpass the same ground location, the solar zenith angles associated   
with the two observations will be slightly different as a result. Second, since the solar zenith   
angle increases from east to west within a swath (i.e., as a result of local solar time increase),   
the solar zenith angle for the same pixels in the overlapping area of two swaths will be   
different due to the tile's relative location difference within the swaths, even if the two   
overlapping swaths were acquired at the same local solar time for the respective swatch   
centers. These points are illustrated in

[Figure. 2](HLS_User_Guide_V2.html#16)

for a tile, 19NGA, near the Equator where

the solar zenith angle changes most dramatically in these cases. It is clear that the mean solar   
zenith angle for a Landsat 8 image on the tile is greater than that for a temporally close   
Sentinel-2 image because Landsat 8 overpasses 30 minutes earlier. There is also temporal   
oscillation in solar zenith angle in the Landsat 8 image time series due to temporally close   
overlapping swaths within a few days. Similarly, the mean solar zenith angle of each   
Sentinel-2 granule in 2019 appears to follow two curves, which are not for S2A and S2B   
separately, but are oscillations from two adjacent orbits. The solar zenith angle used for   
NBAR varies smoothly over time.  
![background image](images/HLS_User_Guide_V2016.png)

*Figure 2. The mean observed solar zenith angle in each tiled Sentinel-2 and Landsat 8 image
and the solar zenith angle used in each image's BRDF normalization, for an equatorial tile
19NGA in 2019. The observed mean solar zenith angle in a Landsat image is higher than
that in a temporally close Sentinel-2 image because of earlier morning overpass time. There
is also day-to-day oscillation in mean observed solar zenith angle for each sensor due to the
alternating observations from two temporally close adjacent orbits.*

4.5. Bandpass Adjustment

The bandpass adjustment in HLS V2.0 is intended to correct for the effects caused by small   
differences in the spectral response functions (RSRs) between Landsat OLI and Sentinel-2   
MSI. The OLI/MSI common spectral bands use the same spectral filters, but there are subtle   
RSR differences; the TOA data calibration minimizes the difference between OLI and MSI   
based on measurements on a few sites, but the difference may be greater over other   
distinctive land cover types. In HLS V2.0, the OLI spectral bandpasses are used as references   
to which the MSI spectral bands are adjusted. The HLS bandpass adjustment is a linear fit   
between equivalent spectral bands simulated from hyperspectral reference data. A total of   
500 hyperspectral spectra from 160 globally distributed EO-1 Hyperion scenes covering   
diverse land cover types (vegetation, desert, urban, and snow) were selected and used to  
![background image](images/HLS_User_Guide_V2017.png)

simulate the MSI and OLI bands by convolution with the respective OLI/MSI RSRs. The   
slope and intercept adjustment coefficients are given in

[Table 5](HLS_User_Guide_V2.html#17)

[.](HLS_User_Guide_V2.html#17)

[](https://www.codecogs.com/eqnedit.php?latex=%5Crho_%7B%5Cscriptsize%20OLI%7D%20%3D%20a%20%5Ctimes%20%5Crho_%7B%5Csmall%7BMSI%7D%7D%20%2B%20b#0)

(5)

The bandpass adjustment is applied to MSI spectral bands that are equivalent to OLI except   
the red-edge, water vapor, and cirrus bands. Separate coefficients are derived for Sentinel 2A,   
2B and 2C. Once corrected, the spectral difference between the MSI and OLI bands is less   
than 2% with a standard deviation of residual error less than 0.005 reflectance units (Claverie   
et al., 2018).   

*Table 5: Bandpass adjustment coefficients*

Sentinel-2A

Sentinel-2B

Sentinel-2C

HLS

Band

name

OLD

Band

number

MSI

Band

number

Slope

(a)

Intercept

(b)

Slope

(a)

Intercept

(b)

Slope

(a)

Intercept

(b)

CA

1

1

0.9959 -0.0002 0.9959 -0.0002

1.003

-0.0

Blue

2

2

0.9778

-0.004

0.9778

-0.004

0.9851 -0.0027

Green

3

3

1.0053 -0.0009 1.0075 -0.0008 1.0038 -0.0009

Red

4

4

0.9765

0.0009

0.9761

0.001

0.9718

0.0011

NIR 1

5

8A

0.9983 -0.0001 0.9966

0

0.9995 -0.0003

SWIR 1

6

11

0.9987 -0.0011

1.000

-0.0003 0.9994 -0.0007

SWIR 2

7

12

1.003

-0.0012 0.9867

0.0004

0.991

0.0004

5. Spatial and Temporal Coverage

HLS V2.0 covers all the global land area except Antarctica, as depicted in a land mask   
(

[Figure](HLS_User_Guide_V2.html#18)

[3](HLS_User_Guide_V2.html#18)

)

derived

from

the

NOAA

shoreline

dataset

(

<https://www.ngdc.noaa.gov/mgg/shorelines/data/gshhg/latest/>

). Antarctica is excluded

because of low solar elevations which compromise the plane-parallel atmospheric correction.   
Note that Landsat and Sentinel-2 sensor data acquisition over some small oceanic islands   
may not be captured regularly. Polar regions located at latitude \>82° are beyond the northern   
reach of the sensors. Image acquisition is not made during the high latitudes winter when the   
sun is too low on the horizon.

The temporal coverage of the HLS V2.0 begins with the first Landsat 8 image in 2013 and   
extends to the present, including data from Landsat 9 from 2021 and Sentinel-2 A/B/C from  
![background image](images/HLS_User_Guide_V2018.png)

2015, 2017 and 2024 respectively. The dataset is generated continuously in a typically with   
data available within 1-2 days since the acquisition.

*Figure 3: HLS V2.0 covers the global land, including major islands but excluding Antarctica.*

<br />

6. Product Formats

6.1. File Format

The HLS 2.0 products are in Cloud Optimized GeoTIFF (COG), one file per data layer to   
offer the flexibility of only downloading the needed data layers and, for cloud-based   
applications, reading the needed spatial subsets within a tile. Each COG file uses internal   
deflate compression and adheres to the COG standard for fast pixel-level access. L30 data are   
stored in directories such as HLS.L30.T17SLU.2020209T155956.V2.0/, which suggests L30   
over tile 17SLU from data acquired on day 209 of 2020 specifically at UTC 155956. This   
example product consists of the following files:

HLS.L30.T17SLU.2020209T155956.V2.0.B01.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.B02.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.B03.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.B04.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.B05.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.B06.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.B07.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.B09.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.B10.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.B11.tif  
![background image](images/HLS_User_Guide_V2019.png)

HLS.L30.T17SLU.2020209T155956.V2.0.Fmask.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.SZA.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.SAA.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.VZA.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.VAA.tif   
HLS.L30.T17SLU.2020209T155956.V2.0.cmr.xml   
HLS.L30.T17SLU.2020209T155956.V2.0.json   
HLS.L30.T17SLU.2020209T155956.V2.0.jpg

<br />

The filenames for individual spectral bands and Fmask cloud mask are self-explanatory.   
Sun zenith angle (SZA), sun azimuth angle (SAA), view zenith angle (VZA) and view   
azimuth angle (VAA) files are also provided (see Section 6.3 for details). File   
HLS.L30.T17SLU.2020209T155956.V2.0.cmr.xml

is

the

metadata

file,

HLS.L30.T17SLU.2020209T155956.V2.0.json contains the size and checksum value of   
each file, and HLS.L30.T17SLU.2020209T155956.V2.0.jpg is a true-color browse   
image.   

The UTC timestamp in the filename is the sensing time at the input Landsat scene center.   
After gridding into the MGRS tiles, it does not accurately indicate the sensing time over   
the tile. If two scenes overlap a MGRS tile, the sensing time of one of the scenes is   
chosen by chance. So, this timing information is not accurate for the MGRS tile center; it   
is intended mainly as a unique identifier, not for quantitative analysis.   

S30 data products are stored in the same format. An example directory   
HLS.S30.T17SLU.2020117T160901.V2.0 may contain the following files:

HLS.S30.T17SLU.2020117T160901.V2.0.B01.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B02.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B03.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B04.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B05.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B06.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B07.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B08.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B8A.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B09.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B10.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B11.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.B12.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.Fmask.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.SZA.tif  
![background image](images/HLS_User_Guide_V2020.png)

HLS.S30.T17SLU.2020117T160901.V2.0.SAA.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.VZA.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.VAA.tif   
HLS.S30.T17SLU.2020117T160901.V2.0.cmr.xml   
HLS.S30.T17SLU.2020117T160901.V2.0.json   
HLS.S30.T17SLU.2020117T160901.V2.0.jpg

The UTC time in the S30 product filenames is the time the sensor begins to sense the sun-lit   
side of the Earth for each orbit, not the exact sensing time over the tile center. When a   
sequence of observations is available on the same day at the high latitude, they can still be   
differentiated by this timing information.

6.2. L30 and S30 Products

The product L30 contains Landsat OLI surface reflectance and TOA TIRS brightness   
temperature gridded at 30 m spatial resolution in MGRS tiles.[](HLS_User_Guide_V2.html#20)

[Table 6](HLS_User_Guide_V2.html#20)

[](HLS_User_Guide_V2.html#20)lists all the data layers

of the L30 product.

*Table 6: All the data layers of the L30 product (SR = Surface Reflectance, NBAR = Nadir
BRDF normalized Reflectance, TOA Refl. = Top of Atmosphere Reflectance, TOA BT = Top
of Atmosphere Brightness temperature).*

**Data layer**

**OLI band**

**number**

**Units**

**Data type**

**Scale**

**Fill value**

**Spatial**

**resolution Description**

B01

1

reflectance

int16

0.0001

-9999

30

NBAR

B02

2

reflectance

int16

0.0001

-9999

30

B03

3

reflectance

int16

0.0001

-9999

30

B04

4

reflectance

int16

0.0001

-9999

30

B05

5

reflectance

int16

0.0001

-9999

30

B06

6

reflectance

int16

0.0001

-9999

30

B07

7

reflectance

int16

0.0001

-9999

30

B09

9

reflectance

int16

0.0001

-9999

30

TOA Refl.

B10

10

degree °C

int16

0.01

-9999

30

TOA BT

B11

11

degree °C

int16

0.01

-9999

30

FMASK

(Table 9)

-

none

uint8

-

255

30

Quality bits  
![background image](images/HLS_User_Guide_V2021.png)

The product S30 contains MSI surface reflectance at 30 m spatial resolution.[](HLS_User_Guide_V2.html#21)

[Table 7](HLS_User_Guide_V2.html#21)

[](HLS_User_Guide_V2.html#21)lists all

the data layers of the S30 product.

*Table 7: All the data layers of the S30 product (SR = Surface Reflectance, NBAR = Nadir
BRDF-Adjusted Reflectance, TOA Refl. = Top of Atmosphere Reflectance).*

Data

Layer

MSI band

number

Units

Data

type

Scalin

g

factor

Fill

value

Spatial

resolution

Description

B01

1

reflectance

int16 0.0001 -9999

30

NBAR

B02

2

reflectance

int16 0.0001 -9999

30

B03

3

reflectance

int16 0.0001 -9999

30

B04

4

reflectance

int16 0.0001 -9999

30

B05

5

reflectance

int16 0.0001 -9999

30

B06

6

reflectance

int16 0.0001 -9999

30

B07

7

reflectance

int16 0.0001 -9999

30

B08

8

reflectance

int16 0.0001 -9999

30

B8A

8A

reflectance

int16 0.0001 -9999

30

B09

9

reflectance

int16 0.0001 -9999

30

TOA Refl

B10

10

reflectance

int16 0.0001 -9999

30

B11

11

reflectance

int16 0.0001 -9999

30

NBAR

B12

12

reflectance

int16 0.0001 -9999

30

FMASK

-

none

uint8

-

255

30

Quality bits

6.3. The Sun And View Angles

HLS V2.0 also provides the sun zenith/azimuth angles (SZA, SAA) and view zenith/azimuth   
angles (VZA, VAA) used in BRDF correction, in case a user may want to do BRDF   
correction differently. The L30 angle data comes with the Collection-2 data; it is originally   
derived for the red band and is taken to represent all bands. The S30 angle data is   
interpolated from the ESA-provided angles in a 5 km grid; HLS selects the view angle of the   
2nd red-edge band and uses it for all bands. The angular files are stored separately as  
![background image](images/HLS_User_Guide_V2022.png)

Cloud-Optimized GeoTIFF layers (SZA, SAA, VZA, VAA) with the attributes summarized   
in

[Table 8](HLS_User_Guide_V2.html#22)

[.](HLS_User_Guide_V2.html#22)

*Table 8: Description of the sun and view angles.*

**Angle bands Units**

**Data type**

**Scaling
factor**

**Fill value**

**Spatial
Resolution**

Sun zenith

degrees

uint16

0.01

40,000

30 m

Sun azimuth

degrees

uint16

0.01

40,000

30 m

View zenith

degrees

uint16

0.01

40,000

30 m

View azimuth degrees

uint16

0.01

40,000

30 m

6.4. Quality Assessment Layer

HLS V2.0 products have one Quality Assessment (QA) layer, based on the Fmask and   
LaSRC output. The Fmask integer output is mapped to an 8-bit (bitwise) representation   
(

[Table 9](HLS_User_Guide_V2.html#22)

) as in HLS V1.4. The HLS processing dilates the Fmask cloud and cloud shadow by

five pixels for L30 and S30 and labels the dilation as "Adjacent to cloud/shadow." The   
qualitative aerosol optical thickness level from LaSRC atmospheric correction is also   
incorporated. The lower six bits in the QA byte may not be mutually exclusive due to the   
possible mixture of categorical labels when data are resampled to 30 m. See Appendix A on   
how to decode the QA bits with simple integer arithmetic.   

*Table 9: Description of the bits in the one-byte Quality Assessment layer. Bits are listed
from the least significant bit (bit 0) to the most significant bit (bit 7).*

<br />

**Bit number**

**Mask name**

**Bit value**

**Mask description**

0

Cirrus

Reserved, but not

used in HLS V2.0

NA

1

Cloud

1

Yes

0

No  
![background image](images/HLS_User_Guide_V2023.png)

6.5. Metadata

Metadata about the L30 or S30 product conform to the NASA Common Metadata   
Repository (CMR) standard and are presented in a file with a cmr.xml filename extension.   
The detailed description of the metadata information is listed in

[Table 11](HLS_User_Guide_V2.html#23)

.

<br />

Table 11: Metadata information for L30 and S30 HLS data products

**L30 Attribute name**

**S30 Attribute name**

**Description**

LANDSAT_PRODUCT_ID

The Landsat 8/9 input L1TP scene product ID. If an   
MGRS tile straddles two adjacent Landsat scenes, two   
product IDs are given.

PRODUCT_URI

The input L1C granule URI. For \~2% of the S30   
products, two partial L1C granules jointly cover the   
tiles; HLS combines the two granules into one, and the   
two input URI are given.

SENSING_TIME

For L30, the WRS scene center sensing time is carried   
over from Landsat L1 metadata.   
For S30, the sensing time at the center of the granule, or

2

Adjacent to

cloud/shadow

1

Yes

0

No

3

Cloud shadow

1

Yes

0

No

4

Snow/ice

1

Yes

0

No

5

Water

1

Yes

0

No

6-7

AOT level

00

Climatology

aerosol

10

Low aerosol

01

Medium aerosol

11

High aerosol  
![background image](images/HLS_User_Guide_V2024.png)

for earlier L1C data, the time at the start of the datatake

HLS_PROCESSING_TIME

The time an L30 or S30 was generated

SPATIAL_COVERAGE

The area percentage of the tile with data

CLOUD_COVERAGE

The percentage of cloud and cloud shadow in   
observation based on Fmask

SPATIAL_RESAMPLING_ALG

For L30, resampling algorithm used in gridding Landsat   
data into the tile.   
For S30, resampling algorithm used in resampling 10   
m/20 m/60 m L1C data to 30 m.

HORIZONTAL_CS_NAME

For L30, the map projection of the input Landsat scene   
or scenes. The UTM zone of the input Landsat scene   
may be different from that of L30.   
For S30, the map projection of the input L1C data, same   
as that of S30.

ULX and ULY

The UTM X/Y coordinate at the upper left corner of the   
tile. A false northing of 0 is used for ULY.

ADD_OFFSET

Value added to the spectral data before they are scaled   
to int16 reflectance data

REF_SCALE_FACTOR

Multiplier to be applied to the int16 reflectance data to   
get the unscaled reflectance

ANG_SCALE_FACTOR

Multiplier to be applied to the uint16 angle bands to get   
the angle in degrees

NCOLS

Number of columns

NROWS

Number of rows

FILLVALUE

Pixel value in the spectral bands where no observation   
was made

QA_FILLVALUE

The QA pixel value where no observation was made

ANG_FILLVALUE

The angle pixel value where no observation was made

MEAN_SUN_AZIMUTH_ANGLE

The mean solar azimuth in the tile

MEAN_SUN_ZENITH_ANGLE

The mean solar zenith in the tile

NBAR_SOLAR_ZENITH

The solar zenith angle used in NBAR derivation

ACCODE

Version of LaSRC used by HLS for S30 or L30

IDENTIFIER_PRODUCT_DOI

L30 or S30 product's DOI

TIRS_SSM_POSITION_ST  
ATUS

Metadata carried over from Landsat L1 data, indicating   
the quality of the thermal data

THERM_SCALE_FACTOR

Multiplier to be applied to the int16 thermal bands to   
get the temperature in Celsius

TIRS_SSM_MODEL

Metadata carried over from Landsat L1 data, indicating   
the quality of the thermal data  
![background image](images/HLS_User_Guide_V2025.png)

PROCESSING_BASELI  
NE

The input Sentinel-2 L1C version number

MSI_BAND_01_BANDP  
ASS_ADJUSTMENT_SL  
OPE_AND_OFFSET

The slope and offset applied to the Sentinel-2 Band   
reflectance in the linear bandpass adjustment

MSI_BAND_02_BANDP  
ASS_ADJUSTMENT_SL  
OPE_AND_OFFSET

MSI_BAND_03_BANDP  
ASS_ADJUSTMENT_SL  
OPE_AND_OFFSET

MSI_BAND_04_BANDP  
ASS_ADJUSTMENT_SL  
OPE_AND_OFFSET

MSI_BAND_11_BANDP  
ASS_ADJUSTMENT_SL  
OPE_AND_OFFSET

MSI_BAND_12_BANDP  
ASS_ADJUSTMENT_SL  
OPE_AND_OFFSET

MSI_BAND_8A_BAND  
PASS_ADJUSTMENT_S  
LOPE_AND_OFFSET

7. Known Issues

There are a few known issues in the HLS V2.0 L30 and S30 products, some concerning the   
science data, and some concerning only the metadata. Some of the known issues have been   
resolved in forward processing, but reprocessing has not been applied to the current HLS   
version. More details about the known issues can be found on the NASA Earthdata HLS   
product pages.

References

Claverie, M., Ju, J., Masek, J.G., Dungan, J.L., Vermote, E.F., Roger, J.-C., Skakun, S.V.,   
\& Justice, C.O. (2018). The Harmonized Landsat and Sentinel-2 surface reflectance data   
set, in press, Remote Sensing of Environment.  
![background image](images/HLS_User_Guide_V2026.png)

Crawford, C. J., Roy, D. P., Arab, S., Barnes, C., Vermote, E., Hulley, G., ... \& Zahn, S.   
(2023). The 50-year Landsat collection 2 archive.

*Science of Remote Sensing*

,

*8*

, 100103.

Drusch, M. et al. (2012) Sentinel-2: ESA's optical high-resolution mission for GMES   
operational services, Remote Sensing of Environment, 120, 25-36.

Drusch, M.; Del Bello, U.; Carlier, S.; Colin, O.; Fernandez, V.; Gascon, F.; Hoersch,   
B.; Isola, C.; Laberinti, P.; Martimort, P.; Meygret, A.; Spoto, F.; Sy, O.; Marchese, F.;   
Bargellini, P. Sentinel-2: ESA's Optical High-Resolution Mission for GMES   
Operational Services. Remote Sensing of Environment 2012, 120, 25--36.

<https://doi.org/10.1016/j.rse.2011.11.026>

.

ESA (2018). Sentinel-2 Data Quality Report S2-PDGS-MPC-DQR.

Irons, J.R., Dwyer, J.L, and J. Barsi (2012) The next Landsat satellite: The Landsat Data   
Continuity Mission, Remote Sensing of Environment, 122, 11-21,   
10.1016/j.rse.2011.08.026.

Ju., [J., Zhou, Q., Freitag, B., Roy, D.P., Zhang, H.K., Sridhar, M., Mandel, J., Arab, S.,
Schmidt, G., Crawford, C.J., Gascon, F., Strobl, P.A., Masek, J.G., Neigh, C.S.R., 2025.
The Harmonized Landsat and Sentinel-2 version 2.0 surface reflectance dataset. Remote
Sens. Environ. 324, 114723.](https://www.zotero.org/google-docs/?CqqIVD)[https://doi.org/10.1016/j.rse.2025.114723.](https://doi.org/10.1016/j.rse.2025.114723)

Li, Z., Zhang, H.K., Roy, D.P., 2018, Investigation of Sentinel-2 bidirectional reflectance   
hot-spot sensing conditions, IEEE Transactions on Geoscience and Remote Sensing,   
10.1109/TGRS.2018.2885967.(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=\&arnumbe  
r=8594675)

Masek, J. G., Vermote, E. F., Saleous, N. E., Wolfe, R., Hall, F. G., Huemmrich, K. F., ...   
\& Lim, T. K.(2006). A Landsat surface reflectance dataset for North America,   
1990-2000. IEEE Geoscience and Remote Sensing Letters, 3(1), 68-72.

Qiu, S., He, B., Zhu, Z., Liao, Z., \& Quan, X. (2017). Improving Fmask cloud and cloud   
shadow detection in mountainous area for Landsats 4--8 images.

*Remote Sensing of*

*Environment*

,

*199*

, 107-119.

Qiu, S., Lin, Y., Shang, R., Zhang, J., Ma, L., \& Zhu, Z. (2018). Making Landsat time   
series consistent: Evaluating and improving Landsat analysis ready data.

*Remote Sensing*

,

*11*

(1), 51.

Qiu S., Zhu Z., and He B., Fmask 4.0: Improved cloud and cloud shadow detection in   
Landsats 4-8 and Sentinel-2 imagery, Remote Sensing of Environment, (2019),

[doi.org/10.1016/j.rse.2019.05.024](http://doi.org/10.1016/j.rse.2019.05.024)

[](http://doi.org/10.1016/j.rse.2019.05.024)  
![background image](images/HLS_User_Guide_V2027.png)

Rengarajan, R., Choate, M., Hasan, M.N., Denevan, A., 2024. Co-registration accuracy   
between Landsat-8 and Sentinel-2 orthorectified products. Remote Sens. Environ. 301,   
113947.[](https://doi.org/10.1016/j.rse.2023.113947)

<https://doi.org/10.1016/j.rse.2023.113947>

[.](https://doi.org/10.1016/j.rse.2023.113947)

Roy, D. P., Li, J., Zhang, H. K., Yan, L., Huang, H., \& Li, Z. (2017). Examination of   
Sentinel-2A multispectral instrument (MSI) reflectance anisotropy and the suitability of a   
general method to normalize MSI reflectance to nadir BRDF adjusted reflectance.   
Remote Sensing of Environment, 199, 25-38.

Roy, D.P., Zhang, H.K., Ju, J., Gomez-Dans, J.L., Lewis, P.E., Schaaf, C.B., Sun, Q., Li,   
J., Huang, H., \& Kovalskyy, V. (2016). A general method to normalize Landsat   
reflectance data to nadir BRDF adjusted reflectance. Remote Sensing of Environment,   
176, 255-271.

Roy, D.P., Li, Z., Zhang, H.K., 2017, Adjustment of Sentinel-2 multi-spectral instrument   
(MSI) red-edge band reflectance to nadir BRDF adjusted reflectance (NBAR) and   
quantification of red-edge band BRDF effects, Remote Sensing, 9(12), 1325.   
(

<http://www.mdpi.com/2072-4292/9/12/1325>

[)](http://www.mdpi.com/2072-4292/9/12/1325)

Schaaf, C. B., Gao, F., Strahler, A. H., Lucht, W., Li, X., Tsang, T., ... \& Lewis, P. (2002).   
First operational BRDF, albedo nadir reflectance products from MODIS. Remote Sensing   
of Environment, 83(1-2), 135-148.

Skakun, S., Roger, J. C., Vermote, E. F., Masek, J. G., \& Justice, C. O. (2017). Automatic   
sub-pixel coregistration of Landsat-8 Operational Land Imager and Sentinel-2A   
Multi-Spectral Instrument images using phase correlation and machine learning based   
mapping. International Journal of Digital Earth, 10(12), 1253-1269.

Skakun, S., Wevers, J., Brockmann, C., Doxani, G., Aleksandrov, M., Batič, M., ... \&   
Žust, L. (2022). Cloud Mask Intercomparison eXercise (CMIX): An evaluation of cloud   
masking algorithms for Landsat 8 and Sentinel-2.

*Remote Sensing of Environment*

,

*274*

,

112990.

Storey, J., Choate, M., \& Lee, K. (2014). Landsat 8 Operational Land Imager On-Orbit   
Geometric Calibration and Performance. Remote Sensing, 6, 11127-11152.

Storey, J., Roy, D. P., Masek, J., Gascon, F., Dwyer, J., \& Choate, M. (2016). A note on   
the temporary misregistration of Landsat-8 Operational Land Imager (OLI) and   
Sentinel-2 Multi Spectral Instrument (MSI) imagery. Remote Sensing of Environment,   
186, 121-122.

Strahler, A.H., Lucht, W., Schaaf, C.B., Tsang, T., Gao, F., Li, X., Lewis, P., \& Barnsley,   
M. (1999). MODIS BRDF/Albedo Product: Algorithm Theoretical Basis Document   
Version 5.0. In M. documentation (Ed.). Boston.  
![background image](images/HLS_User_Guide_V2028.png)

U.S. Geological Survey. (2024).

*Landsat 8--9 Collection 2 Level-2 Science Product Guide*

(LSDS-1619, Version 6.0). In U.S. Department of the Interior documentation (Ed.).   
Reston, VA.

Vermote, E. F., \& Kotchenova, S. (2008). Atmospheric correction for the monitoring of   
land surfaces. Journal of Geophysical Research: Atmospheres, 113(D23).

Vermote, E., Justice, C. O., \& Bréon, F. M. (2009). Towards a generalized approach for   
correction of the BRDF effect in MODIS directional reflectances. IEEE Transactions on   
Geoscience and Remote Sensing, 47(3), 898-908.

Vermote, E., Justice, C., Claverie, M., \& Franch, B. (2016). Preliminary analysis of the   
performance of the Landsat 8/OLI land surface reflectance product. Remote Sensing of   
Environment, 185, 46-56.

Vermote, E., Roger, J. C., Franch, B., \& Skakun, S. (2018). LaSRC (Land Surface   
Reflectance Code): Overview, application and validation using MODIS, VIIRS,   
LANDSAT and Sentinel 2 data's. In

*IGARSS 2018-2018 IEEE International Geoscience*

*and*

*Remote*

*Sensing*

*Symposium*

(pp.

8173-8176).

IEEE.

<https://doi.org/10.1109/IGARSS.2018.8517622>

Yan, L., Roy, D. P., Li, Z., Zhang, H. K., \& Huang, H. (2018). Sentinel-2A   
multi-temporal misregistration characterization and an orbit-based sub-pixel registration   
methodology. Remote Sensing of Environment, 215, 495-506.

Zhang, H.K., Roy, D.P., \& Kovalskyy, V. (2016). Optimal Solar Geometry Definition for   
Global Long- Term Landsat Time-Series Bidirectional Reflectance Normalization. IEEE   
Transactions on Geoscience and Remote Sensing, 54, 1410-1418.

Zhou, Q.; Neigh, C. S. R.; Ju, J.; Dabney, P.; Cook, B.; Zhu, Z.; Crawford, C. J.; Gascon,   
F.; Strobl, P.; Sridhar, M. Toward Seamless Global 30-m Terrestrial Monitoring:   
Evaluating 2022 Cloud Free Coverage of Harmonized Landsat and Sentinel-2 (HLS)   
V2.0.

*IEEE Geoscience and Remote Sensing Letters*

**2025**

,

*22*

, [1--5.](https://doi.org/10.1109/LGRS.2025.3533923)

<https://doi.org/10.1109/LGRS.2025.3533923>

[.](https://doi.org/10.1109/LGRS.2025.3533923)

Zhu, Z., \& Woodcock, C. E. (2012). Object-based cloud and cloud shadow detection in   
Landsat imagery.

*Remote sensing of environment*

,

*118*

, 83-94.

Zhu, Z., Wang, S., \& Woodcock, C.E. (2015). Improvement and expansion of the Fmask   
algorithm: cloud, cloud shadow, and snow detection for Landsats 4-7, 8, and Sentinel 2   
images. Remote Sensing of Environment, 159, 269-277.

![background image](images/HLS_User_Guide_V2029.png)

![background image](images/HLS_User_Guide_V2030.png)

Appendix A: How To Decode The Bit-Packed QA

Quality Assessment (QA) encoded at the bit level provides concise presentation but is   
less convenient for users new to this format. This appendix shows how to decode the QA   
bits with simple integer arithmetic and no explicit bit operation at all. An analogy in the   
decimal system illustrates the idea. For example, given integer 3215, we want to get the   
digit of the hundreds place (i.e., 2). First, divide the integer by 10

2

(i.e., 100) to get an

integer quotient 32, then the digit of the ones place (the least significant digit) of the   
quotient is what we want. To get the ones digit, we compute 32 -- ((32/10) × 10) and get   
2. (Note that integer division 32/10 evaluates to 3, not 3.2.) The same idea applies to   
binary integers. Suppose we get a QA as decimal number 100, which translates into   
binary 01100100, indicating that the aerosol level is low (bits 6-7), it is water (bit 5), and   
adjacent to cloud (bit 2). Suppose we want to find whether it is water by examining the   
value of bit 5. It can be achieved in two steps:

●

Divide 100 by 2

5

to get the quotient, 3, as a result of integer division

●

Find the value of the least significant bit of the quotient by computing 3 -- ((3/2) × 2),   
which is 1. (3/2 = 1 for integer division.)

<br />

The pixel is indeed water-based given the decimal QA value. Note that step 2 above is   
essentially an odd/even number test. When the quotient from step one is odd, the bit in   
question is 1.

*** ** * ** ***

Document Outline
================

* [Harmonized Landsat And Sentinel-2 (HLS) Product User Guide](HLS_User_Guide_V2.html#1)
  * [Table of Contents](HLS_User_Guide_V2.html#2)
  * [](HLS_User_Guide_V2.html#2)
  * [](HLS_User_Guide_V2.html#2)
  * [Acronyms](HLS_User_Guide_V2.html#3)
  * [1.​Introduction](HLS_User_Guide_V2.html#6)
  * [2.​New in V2.0](HLS_User_Guide_V2.html#6)
  * [3.​Products Overview](HLS_User_Guide_V2.html#7)
    * [3.1.​Input Spectral Data](HLS_User_Guide_V2.html#7)
    * [3.2.​Overall HLS Processing Flowchart](HLS_User_Guide_V2.html#9)
    * [3.3.​Products specifications](HLS_User_Guide_V2.html#9)
    * [3.4.​Spectral Bands](HLS_User_Guide_V2.html#10)
    * [3.5.​Output Projection and Gridding](HLS_User_Guide_V2.html#11)
  * [4.​Description of Algorithms](HLS_User_Guide_V2.html#12)
    * [4.1.​Atmospheric Correction](HLS_User_Guide_V2.html#12)
    * [4.2.​Cloud Masking And The Quality Assessment Band](HLS_User_Guide_V2.html#12)
    * [4.3.​Spatial Co-Registration Of Landsat And Sentinel-2 Data](HLS_User_Guide_V2.html#13)
    * [4.4.​View And Illumination Angles Normalization](HLS_User_Guide_V2.html#13)
    * [4.5.​Bandpass Adjustment](HLS_User_Guide_V2.html#16)
  * [5.​Spatial and Temporal Coverage](HLS_User_Guide_V2.html#17)
  * [6.​Product Formats](HLS_User_Guide_V2.html#18)
    * [6.1.​File Format](HLS_User_Guide_V2.html#18)
    * [6.2.​L30 and S30 Products](HLS_User_Guide_V2.html#20)
    * [6.3.​The Sun And View Angles](HLS_User_Guide_V2.html#21)
    * [6.4.​Quality Assessment Layer](HLS_User_Guide_V2.html#22)
    * [](HLS_User_Guide_V2.html#23)
    * [6.5.​Metadata](HLS_User_Guide_V2.html#23)
  * [7.​Known Issues](HLS_User_Guide_V2.html#25)
  * [References](HLS_User_Guide_V2.html#25)
  * [](HLS_User_Guide_V2.html#29)
  * [Appendix A: How To Decode The Bit-Packed QA](HLS_User_Guide_V2.html#30)
  * [](HLS_User_Guide_V2.html#30)
