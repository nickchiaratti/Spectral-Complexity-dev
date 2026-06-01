![background image](images/Planet-Tanager-UserDocumentation001.png)

Tanager Satellite

TANAGER PRODUCT   
SPECIFICATIONS

PLANET.COM

1  
![background image](images/Planet-Tanager-UserDocumentation002.png)

Tanager Product Speciﬁcation

November 2025

TABLE OF CONTENTS

**VERSION TRACKING v0.5**

**1**

[**TABLE OF CONTENTS**](Planet-Tanager-UserDocumentation.html#2)

[**3**](Planet-Tanager-UserDocumentation.html#2)

[**GLOSSARY**](Planet-Tanager-UserDocumentation.html#4)

[**5**](Planet-Tanager-UserDocumentation.html#4)

[**1. OVERVIEW OF DOCUMENT**](Planet-Tanager-UserDocumentation.html#7)

[**8**](Planet-Tanager-UserDocumentation.html#7)

[1.1. COMPANY OVERVIEW](Planet-Tanager-UserDocumentation.html#7)

[8](Planet-Tanager-UserDocumentation.html#7)

[1.2 DATA PRODUCT OVERVIEW](Planet-Tanager-UserDocumentation.html#7)

[8](Planet-Tanager-UserDocumentation.html#7)

[**2. SATELLITE CONSTELLATION AND SENSOR OVERVIEW**](Planet-Tanager-UserDocumentation.html#8)

[**9**](Planet-Tanager-UserDocumentation.html#8)

[2.1 TANAGER SATELLITE CONSTELLATION AND SENSOR CHARACTERISTICS](Planet-Tanager-UserDocumentation.html#8)

[9](Planet-Tanager-UserDocumentation.html#8)

[2.2.1 Collection Modes](Planet-Tanager-UserDocumentation.html#8)

[9](Planet-Tanager-UserDocumentation.html#8)

[Figure 1: Glint Mode](Planet-Tanager-UserDocumentation.html#10)

[11](Planet-Tanager-UserDocumentation.html#10)

[Table 2-A: Tanager Instrumentation \& Operations Overview](Planet-Tanager-UserDocumentation.html#10)

[11](Planet-Tanager-UserDocumentation.html#10)

[**3. TANAGER IMAGERY PRODUCTS**](Planet-Tanager-UserDocumentation.html#11)

[**12**](Planet-Tanager-UserDocumentation.html#11)

[3.1 TANAGER CHUNKING STRATEGY](Planet-Tanager-UserDocumentation.html#11)

[12](Planet-Tanager-UserDocumentation.html#11)

[3.2 TANAGER IMAGERY NAMING CONVENTION](Planet-Tanager-UserDocumentation.html#12)

[13](Planet-Tanager-UserDocumentation.html#12)

[3.3 TANAGER IMAGERY PRODUCT SPECIFICATION](Planet-Tanager-UserDocumentation.html#12)

[13](Planet-Tanager-UserDocumentation.html#12)

[3.3.1 TanagerScene Assets](Planet-Tanager-UserDocumentation.html#12)

[13](Planet-Tanager-UserDocumentation.html#12)

[Table 3-A: Tanager Basic and Ortho Scene Product Components](Planet-Tanager-UserDocumentation.html#12)

[13](Planet-Tanager-UserDocumentation.html#12)

[3.4 TANAGER BASIC SCENE PRODUCT SPECIFICATION](Planet-Tanager-UserDocumentation.html#13)

[14](Planet-Tanager-UserDocumentation.html#13)

[Table 3-B: Tanager Basic Scene Product Attributes](Planet-Tanager-UserDocumentation.html#13)

[14](Planet-Tanager-UserDocumentation.html#13)

[3.5 TANAGER ORTHO SCENES PRODUCT SPECIFICATION](Planet-Tanager-UserDocumentation.html#14)

[15](Planet-Tanager-UserDocumentation.html#14)

[Table 3-C: Tanager Ortho Scene Product Attributes](Planet-Tanager-UserDocumentation.html#15)

[16](Planet-Tanager-UserDocumentation.html#15)

[3.5.1 Tanager Ortho Visual Scene Product Speciﬁcation](Planet-Tanager-UserDocumentation.html#16)

[17](Planet-Tanager-UserDocumentation.html#16)

[Table 3-D: Tanager Visual Ortho Scene Product Attributes](Planet-Tanager-UserDocumentation.html#16)

[17](Planet-Tanager-UserDocumentation.html#16)

[3.6 TANAGER HDF5 PRODUCT COMPONENTS AND FORMAT](Planet-Tanager-UserDocumentation.html#17)

[18](Planet-Tanager-UserDocumentation.html#17)

[Table 3-E: Tanager Basic and Ortho Scene Product Components](Planet-Tanager-UserDocumentation.html#17)

[18](Planet-Tanager-UserDocumentation.html#17)

[Data Fields](Planet-Tanager-UserDocumentation.html#18)

[19](Planet-Tanager-UserDocumentation.html#18)

[Beta Usable Data Masks](Planet-Tanager-UserDocumentation.html#18)

[19](Planet-Tanager-UserDocumentation.html#18)

[Observation Data](Planet-Tanager-UserDocumentation.html#18)

[19](Planet-Tanager-UserDocumentation.html#18)

[Atmospheric Estimates](Planet-Tanager-UserDocumentation.html#19)

[20](Planet-Tanager-UserDocumentation.html#19)

[Spectral Calibration Data](Planet-Tanager-UserDocumentation.html#19)

[20](Planet-Tanager-UserDocumentation.html#19)

[Radiometric Calibration Data](Planet-Tanager-UserDocumentation.html#20)

[21](Planet-Tanager-UserDocumentation.html#20)

[Geolocation Fields](Planet-Tanager-UserDocumentation.html#20)

[21](Planet-Tanager-UserDocumentation.html#20)

[StructMetadata.0](Planet-Tanager-UserDocumentation.html#21)

[22](Planet-Tanager-UserDocumentation.html#21)

[**4. TANAGER DERIVED PRODUCTS**](Planet-Tanager-UserDocumentation.html#24)

[**25**](Planet-Tanager-UserDocumentation.html#24)

[4.1 TANAGER METHANE NAMING CONVENTION](Planet-Tanager-UserDocumentation.html#24)

[25](Planet-Tanager-UserDocumentation.html#24)

[4.2 TANAGER METHANE PRODUCT SPECIFICATION](Planet-Tanager-UserDocumentation.html#24)

[25](Planet-Tanager-UserDocumentation.html#24)

[Table 4-A: Tanager Derived Product Asset-Types](Planet-Tanager-UserDocumentation.html#25)

[26](Planet-Tanager-UserDocumentation.html#25)

[4.2.1 Methane QuickLook (MQL) Product](Planet-Tanager-UserDocumentation.html#25)

[26](Planet-Tanager-UserDocumentation.html#25)

[Table 4-B: Tanager Methane QuickLook Assets](Planet-Tanager-UserDocumentation.html#25)

[26](Planet-Tanager-UserDocumentation.html#25)

[**5. PRODUCT PROCESSING**](Planet-Tanager-UserDocumentation.html#26)

[**27**](Planet-Tanager-UserDocumentation.html#26)

[5.1 TANAGER PROCESSING](Planet-Tanager-UserDocumentation.html#26)

[27](Planet-Tanager-UserDocumentation.html#26)

[Table 5-A: Tanager Processing Steps](Planet-Tanager-UserDocumentation.html#26)

[27](Planet-Tanager-UserDocumentation.html#26)

[5.2 RADIOMETRIC INTERPRETATION](Planet-Tanager-UserDocumentation.html#27)

[28](Planet-Tanager-UserDocumentation.html#27)

PLANET.COM

2  
![background image](images/Planet-Tanager-UserDocumentation003.png)

Tanager Product Speciﬁcation

November 2025

[5.2.1 Radiance Products](Planet-Tanager-UserDocumentation.html#27)

[28](Planet-Tanager-UserDocumentation.html#27)

[5.2.2 Surface Reﬂectance Products](Planet-Tanager-UserDocumentation.html#28)

[29](Planet-Tanager-UserDocumentation.html#28)

[**6. PRODUCT METADATA**](Planet-Tanager-UserDocumentation.html#28)

[**29**](Planet-Tanager-UserDocumentation.html#28)

[6.1 TANAGER ITEM LEVEL METADATA](Planet-Tanager-UserDocumentation.html#28)

[29](Planet-Tanager-UserDocumentation.html#28)

[Table 6-A: Tanager Metadata Schema](Planet-Tanager-UserDocumentation.html#28)

[29](Planet-Tanager-UserDocumentation.html#28)

[6.2 TANAGER DERIVED PRODUCTS](Planet-Tanager-UserDocumentation.html#30)

[31](Planet-Tanager-UserDocumentation.html#30)

[6.2.1 Methane QuickLook](Planet-Tanager-UserDocumentation.html#30)

[31](Planet-Tanager-UserDocumentation.html#30)

[Table 6-B: Tanager Methane Plume Metadata Schema](Planet-Tanager-UserDocumentation.html#30)

[31](Planet-Tanager-UserDocumentation.html#30)

[**APPENDIX A -- IMAGE SUPPORT DATA**](Planet-Tanager-UserDocumentation.html#32)

[**33**](Planet-Tanager-UserDocumentation.html#32)

[BETA USABLE DATA MASK FILE](Planet-Tanager-UserDocumentation.html#32)

[33](Planet-Tanager-UserDocumentation.html#32)

[GEOLOCATION ARRAY](Planet-Tanager-UserDocumentation.html#32)

[33](Planet-Tanager-UserDocumentation.html#32)

No part of this document may be reproduced in any form or any means without the prior written consent of Planet.   
Unauthorized possession or use of this material or disclosure of the proprietary information without the prior written   
consent of Planet may result in legal action. If you are not the intended recipient of this report, you are hereby   
notiﬁed that the use, circulation, quoting, or reproducing of this report is strictly prohibited and may be unlawful.

PLANET.COM

3  
![background image](images/Planet-Tanager-UserDocumentation004.png)

Tanager Product Speciﬁcation

November 2025

**GLOSSARY**

The following list deﬁnes terms used to describe Planet's satellite imagery products. In the event of any   
discrepancy of terms deﬁned in other Planet documents, the terms deﬁned herein relate to this   
Tanager Product Speciﬁcation only.

**Alpha Mask**

An alpha mask is an image channel with binary values that can be used to render areas of the image   
product transparent where no data is available.

**Application Programming Interface (API)**

A set of routines, protocols, and tools for building software applications.

**Atmospheric Correction**

The process of correcting Top-of-Atmosphere (TOA) radiance imagery to account for effects related to   
the intervening atmosphere between the earth's surface and the satellite.

**Beta Usable Data Mask**

The usable data mask is a raster image having the same dimensions as the image product, comprising   
3 bands (Cloud Mask, Cirrus Mask and Surface Water Mask), where each band represents a speciﬁc   
usability class mask.

**Blackﬁll**

Non-imaged pixels or pixels outside of the buffered area of interest that are set to black. They may   
appear as pixels with a value of "0" or as "noData" depending on the viewing software.

**Collect**

A series of scenes captured in a single overpass by a Tanager satellite.

**Circular Error 90th Percentile (CE90)**

CE90 indicates that 90% of all positional measurements will fall within a speciﬁed radius around the true   
position.

**Digital Elevation Model (DEM)**

The representation of continuous elevation values over a topographic surface by a regular array of   
z-values, referenced to a common datum. DEMs are typically used to represent terrain relief.

**GeoJSON**

A standard for encoding geospatial data using JSON (see JSON below).

**GeoTIFF**

An image format with geospatial metadata suitable for use in a GIS or other remote sensing software.

**Ground Sample Distance (GSD)**

The distance between pixel centers, as measured on the ground. It is mathematically calculated based   
on optical characteristics of the telescope, the altitude of the satellite.

PLANET.COM

4  
![background image](images/Planet-Tanager-UserDocumentation005.png)

Tanager Product Speciﬁcation

November 2025

**Graphical User Interface (GUI)**

Web based interfaces enable users to interact with Planet's imagery products without needing   
knowledge of how to use APIs or Application Programming Interfaces.

**HDF-EOS5 (HDF5) image format**

HDF5 ﬁles are self-describing, support partial reading of contents, and contain numerous interrelated   
datasets within a single ﬁle.

**Image Support Data (ISD)**

Additional metadata or ﬁles provided to give context for advanced users.

**Integration Time**

Refers to the duration during which a sensor collects light or radiation from a target to produce a single   
measurement or image pixel, which may inﬂuence the quality of the captured data.

**JavaScript Object Notation (JSON)**

Text-based data interchange format used by the Planet API.

**Metadata**

Data delivered with Planet's imagery products that describes the products content and context and can   
be used to conduct analysis or further processing.

**Nadir**

The point on the ground directly below the satellite.

**Near-Infrared (NIR)**

Near Infrared is a region of the electromagnetic spectrum (780 nm to 1400 nm).

**Orthorectiﬁcation**

The process of removing and correcting geometric image distortions introduced by satellite collection   
geometry, pointing error, and terrain variability.

**Pushbroom**

A remote sensing technique that uses a linear array of detectors to capture continuous cross-track data   
along the ﬂight path of a sensor. As the platform moves, the detectors collect spectral and/or spatial   
information line-by-line, offering efﬁciency without mechanical scanning components.

**Radiometric Correction**

The correction of variations in data that are not caused by the object or image being scanned. These   
include correction for relative radiometric response between detectors, ﬁlling non-responsive detectors   
and scanner inconsistencies.

**Scene**

A single image captured by a Tanager satellite.

PLANET.COM

5  
![background image](images/Planet-Tanager-UserDocumentation006.png)

Tanager Product Speciﬁcation

November 2025

**Sensor Correction**

The correction of variations in the data that are caused by sensor geometry, attitude, and ephemeris.

**Shortwave-Infrared (SWIR)**

Shortwave Infrared is a region of the electromagnetic spectrum (1400 nm to 3000 nm).

**Sun Azimuth**

The angle of the sun as seen by an observer located at the target point, as measured in a clockwise   
direction from true north.

**Sun Elevation**

The angle of the sun above the horizon.

**Sun Synchronous Orbit (SSO)**

A geocentric orbit that combines altitude and inclination in such a way that the satellite passes over any   
given point of the planet's surface at the same local solar time.

**Surface Reﬂectance (SR)**

Surface reﬂectance is the proportion of light reﬂected by the surface of the earth. It is a ratio of surface   
radiance to surface irradiance, and as such is unitless, and typically has values between 0 and 1.

PLANET.COM

6  
![background image](images/Planet-Tanager-UserDocumentation007.png)

Tanager Product Speciﬁcation

November 2025

1. OVERVIEW OF DOCUMENT

This document describes Planet's hyperspectral satellite imagery products generated from the current   
constellation of Tanager satellites. It is intended for users of satellite imagery interested in working with   
Tanager's product offerings.

1.1. COMPANY OVERVIEW

Planet uses an agile aerospace approach for the design of its satellites, mission control, and operations   
systems; and the development of its web-based platform for imagery processing and delivery.

Planet was founded with the mission to image the Earth every day and make change visible, accessible,   
and actionable. Over the past decade with its customers, Planet has revolutionized the Earth   
observation industry, democratizing access to satellite data beyond the traditional agriculture and   
defense sectors.

To that end, Planet offers Planet Insights Platform, a cloud-native platform that combines a near-daily   
scan of the Earth's landmass and strategic waterways with tools for advanced analysis to derive insights   
and make timely, informed decisions. Planet Insights Platform also provides access to high-resolution   
imagery, derived data products, and the infrastructure for users to develop web applications, integrate   
with GIS workﬂows, and drive data science and machine learning applications.

Businesses, governments, and research institutions leverage Planet's data and platform to scale their   
operations, increase efﬁciency, mitigate risk, and develop novel solutions to address their most pressing   
challenges. This helps them stay ahead in ever-changing global contexts and ultimately capture   
unforeseen windows of opportunity.

1.2 DATA PRODUCT OVERVIEW

Planet operates the PlanetScope, SkySat, Pelican, and Tanager Earth-imaging constellations. Imagery is   
collected and processed in a variety of formats to serve different use cases, such as mapping, deep   
learning, disaster response, precision agriculture, or simple temporal image analytics to create rich   
information products.

Tanager is an imaging spectrometer that operates as a line scanner collecting approximately 426 bands   
over the spectral range of 376 to 2500 nm (\~0.4 to \~2.5 µm) with a spectral sampling of about \~5 nm.   
Planet generates calibrated radiance data from raw satellite measurements and further processes   
radiance to surface reﬂectance.

Planet offers two geometry types for TanagerScene imagery: Basic and Ortho. Tanager Basic Scene   
products contain the hyperspectral data (e.g., radiance or surface reﬂectance) in image space that has   
been processed to remove distortions caused by terrain and sensor and to provide precise geolocation

PLANET.COM

7  
![background image](images/Planet-Tanager-UserDocumentation008.png)

Tanager Product Speciﬁcation

November 2025

information in geographic coordinate space. This geolocation information is provided as a separate   
array (longitude and latitude) to enable users to orthorectify or project the product themselves. An   
Ortho Scene product is orthorectiﬁed and the product was designed for a wide variety of applications   
that require imagery with an accurate geolocation and cartographic projection. It has been processed   
to remove distortions caused by terrain and can be used for cartographic purposes. The Ortho Scenes   
are delivered as visual (RGB), top-of-atmosphere radiance and surface reﬂectance products. Ortho   
Scenes are radiometrically-, sensor-, and geometrically-corrected products that are projected to a   
cartographic map projection. The native ﬁle format for both Tanager's Basic and Ortho assets is   
HDF-EOS5 (HDF5) image format. HDF5 ﬁles are self-describing, support partial reading of contents, and   
contain numerous interrelated datasets within a single ﬁle. Inside an HDF5 ﬁle, there are groups,   
datasets, and attributes. Groups are like folders and can hold other groups or datasets. Attributes are   
details about the groups or datasets. One can add attributes to anything in the ﬁle. Datasets are   
collections of data that are arranged like an array. Users can ﬁnd the HDF5 documentation here:

<https://www.hdfgroup.org/solutions/hdf5/>

[.](https://www.hdfgroup.org/solutions/hdf5/)

In addition to imagery, Planet also provides Derived Products based on Tanager collections. The ﬁrst   
Derived Product is a methane detection product, TanagerMethane item-type, that enables leak   
detection, regulatory enforcement, and emissions inventory applications, among others.

2. SATELLITE CONSTELLATION AND SENSOR OVERVIEW

2.1 TANAGER SATELLITE CONSTELLATION AND SENSOR CHARACTERISTICS

The Tanager satellite is an imaging spectrometer capturing approximately 426 Bands with \~5 nm   
spacing between an approximate range of 380-2500 nm, with the ﬁrst launched in August 2024. Planet   
intends to launch additional Tanager satellites to meet market demands over the coming years.

Each satellite is 3-axis stabilized and agile enough to slew between different targets of interest. Each   
satellite has a single electric propulsion (EP) thruster for orbital control, along with four reaction wheels   
and three magnetic torquers for attitude control.

All Tanagers contain three-mirror anastigmat (TMA) telescopes with a focal length of 400 mm and   
Dyson form spectrometers, with a 640x480 pixel Mercury-Cadmium-Telluride (MCT) detector as the   
focal plane array.

2.2.1 Collection Modes

Due to the minimum integration time of 8 ms, the lower bound of the along-track length of each pixel   
is currently at around 58 m when scanning at the orbital rate in pushbroom mode. This along-track   
elongation of area registered at each sensing element on the focal plane array (FPA) causes pixels in the   
geometrically corrected images to be rectangular, and for most use cases this pixel stretching is   
generally an undesirable effect.

PLANET.COM

8  
![background image](images/Planet-Tanager-UserDocumentation009.png)

Tanager Product Speciﬁcation

November 2025

In order to overcome this elongation effect and achieve signal-to-noise ratio (SNR) goals, Tanager   
performs attitude maneuvering, referred to as 'back nodding,' to be able to get a time-prolonged   
capture over an area compared to what a simple pushbroom scan would allow at the nominal ground   
scan speed. In other words, the imaging starts before the satellite ﬂies above the starting point of the   
imaging ground track, and ﬁnishes after it has passed over the ending point of the imaging ground   
track. This strategy allows for more imaging time to be devoted to a potentially smaller area, thus   
yielding a higher SNR.

Tanager imaging modes. For each mode, the Tanager satellite back-nods with a given angular rate over

some distance to provide ground motion compensation.

Standard Sensitivity (1x8ms, 8ms): In this mode, back-nodding is used to slow down the ground scan   
rate just enough to prevent pixel stretching, maintaining square pixels while maximizing swath length.   
The primary goal is to avoid elongation without speciﬁcally targeting higher SNR.

Medium Sensitivity (2x8ms, 16ms): In this mode, the ground scanning rate is slowed further to provide   
an effective integration time of approximately 16 ms. The result of this increased effective integration   
time is an \~1.4x increase in the SNR over Standard Sensitivity mode.

High Sensitivity (3x8ms, 24ms): In this mode, the ground scanning rate is slowed further to provide an   
effective integration time of approximately 24 ms. The result of this increased effective integration time   
is an \~1.7x increase in the SNR over Standard Sensitivity mode.

Maximum Sensitivity (4x8ms, 32ms): This mode employs much faster back-nodding to signiﬁcantly slow   
the ground scan rate and achieve the highest possible SNR, with an effective integration time of   
approximately 32 ms. The result of this increased effective integration time is an \~2x increase in SNR   
over Standard Sensitivity mode.

Glint mode imaging which is in public beta as of November 2025 is essential for monitoring methane   
plumes over dark ocean surfaces. The strategy in this imaging mode is to maximize the background   
photon count from the ocean surface, which is usually low, while observing a target of interest on the   
ocean's surface. The satellite will point towards the sun's specular reﬂection point on the ocean, where   
the sunlight reﬂects off of the surface at the same angle that a satellite is viewing the surface, while the   
image also contains a target of interest such as an oil rig platform or a large ship. Tanager, the sun's   
specular reﬂection point on the ocean, and the Sun itself have to be roughly in the same plane. There is

PLANET.COM

9  
![background image](images/Planet-Tanager-UserDocumentation010.png)

Tanager Product Speciﬁcation

November 2025

some tolerance on how far out the specular point can be from that ideal plane. For a sun synchronous   
orbit with LTAN near noon, the sensor will point ahead and then behind in the orbit during each pass.

A notional representation of this imaging mode is shown in the ﬁgure below.

Figure 1: Glint Mode

Table 2-A: Tanager Instrumentation \& Operations Overview

Attribute

Value

Instrument

Tanager Vehicle

Sensor Type

Dyson-type imaging spectrometer

Telescope

Three-mirror anastigmat (TMA) telescope with a 22 cm aperture

Orbital Parameters (Altitude \&   
Inclination)

Tanager-1: currently 430 km at 97.5°

Min/Max Latitude Coverage

± 90°

Equatorial Crossing (local solar time)

11:00-13:00

Spectral Range

Tanager-1: 376-2500 nm   
Tanager-1: \~5 nm spacing, 5-7 nm Full Width at Half Maximum

Ground Sample Distance

Tanager-1: \~32 m at nadir (from 430 km altitude)

Off-Nadir Angle

± 30° along track and/or cross-track

Maximum Image Strip per task

Determined based on sensitivity mode:   
Standard (1x8ms): 390 km   
Medium (2x8ms): 263 km   
High (3x8ms): 172 km

PLANET.COM

10  
![background image](images/Planet-Tanager-UserDocumentation011.png)

Tanager Product Speciﬁcation

November 2025

Maximum (4x8ms): 127 km

Revisit Time (per satellite)

Approx. weekly depending on latitude

Instrument Duty Cycle

Approx. 8%

Image Capture Capacity

300,000 km²/day

Geolocation Knowledge

\< 1 pixel at CE90

Availability Date (In Operations)

Sept 2024 - Present

3. TANAGER IMAGERY PRODUCTS

Tanager imagery products are available as either individual Basic or Ortho Scenes and radiance or   
surface reﬂectance. These products can be obtained from the Planet API through the TanagerScene   
item type.

3.1 TANAGER CHUNKING STRATEGY

A single Tanager collection can be quite long and in order to process and deliver Tanager assets with   
more reasonable ﬁle sizes, the collections will be dynamically chunked. Each collect will be separated   
into TanagerScenes based on a set of rules:

1. A scene is at most 750 lines long.   
2. A scene is at least 325 lines long.   
3. The strategy seeks to produce square-ish scenes.   
4. All scenes within a single collection are approximately the same size. The worst case scenario is

that a collect will have only two different scene sizes. These two sizes will vary by only a single   
line.

Example chunking below. Each

is a TanagerScene.

1700 line collect:

0:567

567:1134

1134:1700

2600 line collect:

0:650

650:1300

1300:1950 1950:2600

PLANET.COM

11  
![background image](images/Planet-Tanager-UserDocumentation012.png)

Tanager Product Speciﬁcation

November 2025

3.2 TANAGER IMAGERY NAMING CONVENTION

The name of each acquired Tanager image is designed to be unique and allow for easier recognition   
and sorting of the imagery. It includes the date and time of capture, as well as the id of the satellite that   
captured it. The name of each downloaded image product is composed of the following elements:

TanagerScene

:

\<acquisition date\>_\<acquisition time\>_\<hundredths of a second\>_\<satellite_id\>_\<asset_type\> .\<extension\>

Example: 20241006_154116_92_4001_ortho_radiance_hdf5.h5   
Searchable product in Data API is: TanagerScene 20241006_154116_92_4001

3.3 TANAGER IMAGERY PRODUCT SPECIFICATION

3.3.1 TanagerScene Assets

The following products are generated for each

TanagerScene

item published to the Planet catalog. Find

their asset-type name and description below.

Table 3-A: Tanager Basic and Ortho Scene Product Components

Item-Type

Asset-Type

Description

**TanagerScene**

Ortho Radiance Scene

ortho_radiance_hdf5

Orthorectiﬁed, Top of atmosphere radiance (at sensor)   
calibrated, in HDF5 format.

Basic Radiance Scene

basic_radiance_hdf5

Unorthorectiﬁed, Top of atmosphere radiance (at sensor)   
calibrated, in HDF5 format. Not projected to a cartographic   
projection.

Ortho Surface Reﬂectance   
Scene

ortho_sr_hdf5

Orthorectiﬁed, atmospherically corrected surface   
reﬂectance product, in HDF5 format.

Basic Surface Reﬂectance   
Scene

basic_sr_hdf5

Unorthorectiﬁed, atmospherically corrected surface   
reﬂectance product, in HDF5 format. Not projected to a   
cartographic projection.

Ortho Visual Scene

ortho_visual

Orthorectiﬁed red, green, blue (RGB) visual image with   
color-correction, in GeoTIFF format.

Ortho Beta Usable Data   
Mask (UDM)

ortho_beta_udm

Orthorectiﬁed usable data mask (in beta), in GeoTIFF   
format.

Basic Beta UDM

basic_beta_udm

Unorthorectiﬁed usable data mask (in beta), in GeoTIFF   
format

PLANET.COM

12  
![background image](images/Planet-Tanager-UserDocumentation013.png)

Tanager Product Speciﬁcation

November 2025

Geolocation Array

geolocation_array

Longitudes and Latitudes in WGS84 of centers of pixels, in   
GeoTIFF format.

3.4 TANAGER BASIC SCENE PRODUCT SPECIFICATION

Tanager Basic Scene products contain the hyperspectral data (e.g., radiance or surface reﬂectance) in   
image space that has been processed to remove distortions caused by terrain and sensor and to provide   
precise geolocation information in geographic coordinate space. This geolocation information is   
provided as a separate array (longitude and latitude) to enable users to orthorectify or project the   
product themselves. This product line is available in HDF-EOS5 format.

This product line is available in HDF-EOS5 format. Tanager Basic Scene products contain the   
hyperspectral data (e.g., radiance or surface reﬂectance) in image space that has been processed to   
remove distortions caused by terrain and sensor and to provide precise geolocation information in   
geographic coordinate space. This geolocation information is provided as a separate array (longitude   
and latitude) to enable users to orthorectify or project the product themselves.

See[](Planet-Tanager-UserDocumentation.html#26)

[Section 5.1 - Tanager Processing](Planet-Tanager-UserDocumentation.html#26)

for more details. The table below describes the attributes for the

Tanager Basic Scene product:

Table 3-B: Tanager Basic Scene Product Attributes

TANAGER BASIC SCENE PRODUCT ATTRIBUTES

**Information Content**

Image Conﬁgurations

Approximately 426 channel radiance product

Product Framing

Scene based, produced by a line scanner.

One scene has the nominal dimensions:

●

Width: approximately 600px

●

Length: variable by size of collect and Planet chunking. See

[section 3.1](Planet-Tanager-UserDocumentation.html#11)

[Tanager Chunking Strategy.](Planet-Tanager-UserDocumentation.html#11)

●

Depth: approximately 426px

Spectral Bands

Approximate range of 376 - 2500 nm with \~5 nm spacing between channels

Ground Sample Distance (GSD)

Tanager-1: \~32 m at nadir (from 430 km altitude)

**Processing**

Pixel Size

N/A

Radiometric Corrections

●

Dark bias removed

PLANET.COM

13  
![background image](images/Planet-Tanager-UserDocumentation014.png)

Tanager Product Speciﬁcation

November 2025

●

Pedestal shift corrected

●

Flat ﬁled applied

●

Bad pixels corrected

●

Spatial/spectral stray light corrected

●

Optical ghosts corrected

●

Conversion to absolute radiometric values based on calibration coefﬁcients

●

Order sorting ﬁlter seams interpolated

Geometric Corrections

Sensor-related effects are corrected using sensor telemetry and a sensor model.   
Orthorectiﬁcation uses (GCPs) ground control points and ﬁne (DEMs) digital   
elevation models (10 m to 90 m posting).

Atmospheric Corrections/Estimates

Relevant for surface reﬂectance products only.

●

Estimated surface reﬂectance, water vapor concentration, and aerosol optical   
thickness at 550 nm and their respective uncertainties derived with the Imaging   
Spectrometer Optimal FITting (ISOFIT) model that utilizes various radiative   
transfer models and statistical description of surface, instrument and   
atmosphere.

Bit Depth

TOA Radiance - W m-2 sr-1 μm-1: 32-bit ﬂoating point   
Surface Reﬂectance - Unitless: 32-bit ﬂoating point

Resampling Kernel

N/A

Map Projection

N/A

Geolocation Accuracy

At 30m GSD \<50 m absolute CE90, \<25 m relative CE90 where georectiﬁcation   
succeeds. In other words, at 30m GSD \<2 pixels absolute CE90 and \<1 pixel relative   
CE90 where georectiﬁcation succeeds.

Horizontal Datum

WGS84

3.5 TANAGER ORTHO SCENES PRODUCT SPECIFICATION

The Tanager Ortho Scene product is orthorectiﬁed and the product was designed for a wide variety of   
applications that require imagery with an accurate geolocation and cartographic projection. It has been   
processed to remove distortions caused by terrain and can be used for cartographic purposes. The   
Ortho Scenes are delivered as visual (RGB), top-of-atmosphere radiance and surface reﬂectance   
products. Ortho Scenes are radiometrically-, sensor-, and geometrically-corrected products that are   
projected to a cartographic map projection. The geometric correction uses ﬁne Digital Elevation Models   
(DEMs) with a post spacing of between 10 and 90 meters.

Ground Control Points (GCPs) are used in the creation of every image and the accuracy of the product   
will vary from region to region based on available GCPs. Computer vision algorithms are used for   
extracting feature points such as OpenCV's STAR keypoint detector and FREAK keypoint extractor. The   
GCP and tiepoint matching is done using a combination of RANSAC, phase correlation and mutual   
information.

PLANET.COM

14  
![background image](images/Planet-Tanager-UserDocumentation015.png)

Tanager Product Speciﬁcation

November 2025

The table below describes the attributes for the Tanager Ortho Scene product:

Table 3-C: Tanager Ortho Scene Product Attributes

TANAGER ORTHO SCENE PRODUCT ATTRIBUTES

**Information Content**

Image Conﬁgurations

Approximately 426 bands

Product Framing

Scene based, produced by a line scanner.

One scene has the nominal dimensions:

●

Width: approximately 600px

●

Length: variable by size of collect and Planet chunking. See

[section 3.1](Planet-Tanager-UserDocumentation.html#11)

[Tanager Chunking Strategy.](Planet-Tanager-UserDocumentation.html#11)

●

Depth: approximately 426px

Spectral Bands

Approximate range of 376 - 2500 nm with \~5 nm spacing between channels

Ground Sample Distance (GSD)

30m

**Processing**

Pixel Size

30m

Radiometric Corrections

●

Dark bias removed

●

Pedestal shift corrected

●

Flat ﬁled applied

●

Bad pixels corrected

●

Spatial/spectral stray light corrected

●

Optical ghosts corrected

●

Conversion to absolute radiometric values based on calibration coefﬁcients

●

Order sorting ﬁlter seams interpolated

Geometric Corrections

Sensor-related effects are corrected using sensor telemetry and a sensor model.   
Orthorectiﬁcation uses (GCPs) ground control points and ﬁne (DEMs) digital   
elevation models (10 m to 90 m posting).

Atmospheric Corrections/Estimates Relevant for surface reﬂectance products only.

●

Estimated surface reﬂectance, water vapor concentration, and aerosol   
optical thickness at 550 nm and their respective uncertainties derived with   
the Imaging Spectrometer Optimal FITting (ISOFIT) model that utilizes   
various radiative transfer models and statistical description of surface,   
instrument and atmosphere.

Bit Depth

TOA Radiance - W m-2 sr-1 μm-1: 32-bit ﬂoating point   
Surface Reﬂectance - Unitless: 32-bit ﬂoating point

Resampling Kernel

Nearest Neighbor

PLANET.COM

15  
![background image](images/Planet-Tanager-UserDocumentation016.png)

Tanager Product Speciﬁcation

November 2025

Map Projection

UTM

Geolocation Accuracy

At 30m GSD \<50 m absolute CE90, \<25 m relative CE90 where georectiﬁcation   
succeeds.

Horizontal Datum

WGS84

3.5.1 Tanager Ortho Visual Scene Product Speciﬁcation

The Tanager Visual Ortho Scene product is orthorectiﬁed and color-corrected. This correction attempts   
to optimize colors as seen by the human eye providing images as they would look if viewed from the   
perspective of the satellite. This product has been processed to remove distortions caused by terrain   
and can be used for cartographic mapping and visualization purposes. This correction also eliminates   
the perspective effect on the ground (not on buildings), restoring the geometry of a vertical shot.

The Visual Ortho Scene product is optimal for simple and direct use of an image. It is designed and   
made visually appealing for a wide variety of applications that require imagery with an accurate   
geolocation and cartographic projection. The product can be used and ingested directly into a   
Geographic Information System.

Table 3-D: Tanager Visual Ortho Scene Product Attributes

TANAGER VISUAL ORTHO SCENE PRODUCT ATTRIBUTES

Product Attribute

Description

Product Components and Format

Tanager Visual Ortho Scene product consists of the following ﬁle components:

●

Image File -- GeoTIFF format

**Information Content**

Image Conﬁgurations

3-band natural color

●

red band (665 nm),

●

green band (565 nm),

●

blue band (490 nm)

Product Framing

Scene based, produced by a line scanner.

One scene has the nominal dimensions:

●

Width: approximately 600px

●

Length: variable by size of collect and Planet chunking. See[](Planet-Tanager-UserDocumentation.html#11)

[section 3.1](Planet-Tanager-UserDocumentation.html#11)

[Tanager Chunking Strategy.](Planet-Tanager-UserDocumentation.html#11)

●

Depth: approximately 426px

Ground Sample Distance (GSD)

30m

PLANET.COM

16  
![background image](images/Planet-Tanager-UserDocumentation017.png)

Tanager Product Speciﬁcation

November 2025

**Processing**

Pixel Size

30m

Geometric Corrections

Sensor-related effects are corrected using sensor telemetry and a sensor model.   
Orthorectiﬁcation uses (GCPs) ground control points and ﬁne (DEMs) digital elevation   
models (10 m to 90 m posting).

Bit Depth

8-bit

Resampling Kernel

Cubic

Map Projection

UTM

Geolocation Accuracy

At 30m GSD \<50 m absolute CE90, \<25 m relative CE90 where georectiﬁcation   
succeeds

Horizontal Datum

WGS84

3.6 TANAGER HDF5 PRODUCT COMPONENTS AND FORMAT

The following data is included in the HDF5 ﬁle structure. Be sure to notice which products the data is   
and is not produced for. For example, Atmospheric Estimate data is not included in the radiance   
product.

Table 3-E: Tanager Basic and Ortho Scene Product Components

Product Attribute

Description

Product Components and Format

The Tanager Basic Scene product provided in HDF5

1

format consists of the following

ﬁle components:

A "HDFEOS" group containing data complying with version 1.1 of the HDF-EOS5

2

spec

with the following groups:

●

HDFEOS/SWATHS/HYP/Data Fields

●

HDFEOS/SWATHS/HYP/Geolocation Fields

●

HDFEOS INFORMATION/StructMetadata.0

The

*HDFEOS/SWATHS/HYP*

and

*HDFEOS/GRIDS/HYP*

have the attribute

*strip_id*

to relate the TanagerScene to the

strip it was derived from.

2

HDF-EOS5 Data Model, File Format and Library, May, 2016, Version 1.1,

<https://www.earthdata.nasa.gov/s3fs-public/imported/ESDS-RFC-008-v1.1.pdf>   
<https://www.hdfeos.org/>

1

<https://www.hdfgroup.org/>

PLANET.COM

17  
![background image](images/Planet-Tanager-UserDocumentation018.png)

Tanager Product Speciﬁcation

November 2025

Data Fields

Beta Usable Data Masks

Listed in the

*HDFEOS/SWATHS/HYP/Data Fields*

group of the HDF-EOS5 ﬁles. Also generated as a

geoTIFF asset named ortho_beta_udm (see Appendix A). The Tanager beta usable data mask is based   
on the EMIT codebase (described by [Sandford et al. 2020](https://amt.copernicus.org/articles/13/7047/2020/)

[3](https://amt.copernicus.org/articles/13/7047/2020/)

and [Thompson et al. 2014](https://ieeexplore.ieee.org/document/6744616)

[4](https://ieeexplore.ieee.org/document/6744616)

).

Dataset

Description

beta_cloud_mask

Binary indicating pixels which are clear ("1" indicates cloud)

beta_cirrus_mask

Binary indicating pixels in which cirrus clouds are identiﬁed ("1" indicates cirrus)

nodata_pixels

Binary indicating pixels with no data ("1" indicates no data)

Observation Data

Per-pixel metadata describing observation parameters. Listed in the

*HDFEOS/SWATHS/HYP/Data Fields*

group of the HDF-EOS5 ﬁles.

Dataset

Description

sensor_to_ground_path_length

Distance from the pixel on the sensor to corresponding   
pixel on the ground in meters

sensor_zenith

Angle from local zenith at the ground pixel to the pixel on   
the sensor in decimal degrees

sensor_azimuth

Angle from true North at the ground pixel to the pixel on   
the sensor in decimal degrees

sun_zenith

Angle from local zenith at the ground pixel to the sun in   
decimal degrees

sun_azimuth

Angle from true North at the ground pixel to the sun in   
decimal degrees

time

Acquisition time for each pixel in UTC. Time of acquisition   
is part of the Geolocations Fields for basic products and   
part of the per-pixel metadata for ortho products.

4

D. R. Thompson et al., "Rapid Spectral Cloud Screening Onboard Aircraft and Spacecraft," in IEEE Transactions on Geoscience and

Remote Sensing, vol. 52, no. 11, pp. 6779-6792, Nov. 2014, doi: 10.1109/TGRS.2014.2302587.

3

Sandford, M. W., Thompson, D. R., Green, R. O., Kahn, B. H., Vitulli, R., Chien, S., Yelamanchili, A., and Olson-Duvall, W.: Global cloud

property models for real-time triage on board visible--shortwave infrared spectrometers, Atmos. Meas. Tech., 13, 7047--7057,   
https://doi.org/10.5194/amt-13-7047-2020, 2020.

PLANET.COM

18  
![background image](images/Planet-Tanager-UserDocumentation019.png)

Tanager Product Speciﬁcation

November 2025

Atmospheric Estimates

Only produced for surface reﬂectance products. Pixel-level atmospheric estimates as an output of the   
reﬂectance retrieval. Listed in the

*HDFEOS/SWATHS/HYP/Data Fields*

group of the HDF-EOS5 ﬁles.

Dataset

Description

column_water_vapor

Column water vapor as estimated during the reﬂectance   
retrieval in g/cm².

aerosol_optical_depth

Aerosol optical depth at 550 nm (AOD550) as estimated   
during the reﬂectance retrieval (unitless).

Spectral Calibration Data

Per-band metadata describing the spectral calibration data of the sensor. Stored as attributes to

*HDFEOS/SWATHS/HYP/Data Fields/toa_radiance*

or

*HDFEOS/SWATHS/HYP/Data*

*Fields/surface_reﬂectance*

dataset.

Dataset

Description

wavelengths

The center wavelength of the spectral response for each band (nm)

PLANET.COM

19  
![background image](images/Planet-Tanager-UserDocumentation020.png)

Tanager Product Speciﬁcation

November 2025

fwhm

The width, in nanometers, of a spectral curve at half the maximum amplitude

good_wavelengths\*

Binary to indicate which bands do not accurately represent surface reﬂectance   
due to water absorption of other atmospheric features. Good wavelengths are   
indicated by 1, bad wavelengths are 0.

\*

*Only produced for Surface Reﬂectance products.*

Radiometric Calibration Data

Per-band metadata describing radiometric calibration coefﬁcients used by Planet. Stored as attributes   
to

*HDFEOS/SWATHS/HYP/Data Fields/toa_radiance*

.

Dataset

Description

applied_radiometric_coefﬁcients

Radiometric calibration coefﬁcients that were used to convert DNs to TOA   
Radiance for each band, in units of W/(m²\*sr\*μm)

Geolocation Fields

Input geometry to be used for georeferencing of the data. The

*Geolocation Fields*

group has the

attribute

*Planet_Ortho_Framing*

. This is only produced for Basic Products.

Datasets

Description

Latitude

WGS-84 latitude in decimal degrees

Longitude

WGS-84 longitude in decimal degrees

Time

Acquisition time per line in UTC. Unix epoch is the number of seconds that have elapsed since   
January 1, 1970 midnight UTC/GMT.

A

*Planet_Ortho_Framing*

attribute is a JSON encoded string containing information about the ortho

framing of a TanagerScene. While Planet does not offer a TanagerCollect, customers can use this   
information to reconstruct ortho collect products.

PLANET.COM

20  
![background image](images/Planet-Tanager-UserDocumentation021.png)

JavaScript

None

Tanager Product Speciﬁcation

November 2025

{

"epsg_code"

:

32613

,

"rows"

:

608

,

"cols"

:

1016

,

"geotransform"

: \[

503220

.

0

,

30

.

0

,

0

.

0

,

4410990

.

0

,

0

.

0

,

-30

.

0

\]

}

*epsg_code*

,

*rows*

,

*cols*

, and

*geotransform*

describe framing information for a TanagerScene ortho

product. For a given collect,

*rows*

,

*cols*

, and

*geotransform*

may be different for each TanagerScene

belonging to that TanagerCollect.

StructMetadata.0

StructMetadata.0 is based on the HDF-EOS5 standards and contains general information about the   
structure of the ﬁle, dataset dimensions, compression, data types and georeferencing information.

GROUP=SwathStructure   
END_GROUP=SwathStructure   
GROUP=GridStructure

GROUP=GRID_1

GridName="HYP"   
Band=426   
XDim=846   
YDim=753   
UpperLeftPointMtrs=(570300.00,3555450.00)   
LowerRightMtrs=(595680.00,3532860.00)   
Projection=HE5_GCTP_UTM   
ZoneCode=13   
SphereCode=12   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4   
PixelRegistration=HE5_HDFE_CORNER   
GridOrigin=HE5_HDFE_GD_UL   
GROUP=Dimension

OBJECT=Dimension_1

DimensionName="Band"   
Size=426

END_OBJECT=Dimension_1   
OBJECT=Dimension_2

DimensionName="YDim"   
Size=753

END_OBJECT=Dimension_2   
OBJECT=Dimension_3

DimensionName="XDim"

PLANET.COM

21  
![background image](images/Planet-Tanager-UserDocumentation022.png)

Tanager Product Speciﬁcation

November 2025

Size=846

END_OBJECT=Dimension_3

END_GROUP=Dimension   
GROUP=DataField

OBJECT=DataField_1

DataFieldName="toa_radiance"   
DataType=H5T_NATIVE_FLOAT   
DimList=("Band","YDim","XDim")   
MaxdimList=("Band","YDim","XDim")   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4

END_OBJECT=DataField_1   
OBJECT=DataField_2

DataFieldName="sensor_zenith"   
DataType=H5T_NATIVE_FLOAT   
DimList=("YDim","XDim")   
MaxdimList=("YDim","XDim")   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4

END_OBJECT=DataField_2   
OBJECT=DataField_3

DataFieldName="sensor_azimuth"   
DataType=H5T_NATIVE_FLOAT   
DimList=("YDim","XDim")   
MaxdimList=("YDim","XDim")   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4

END_OBJECT=DataField_3   
OBJECT=DataField_4

DataFieldName="sensor_to_ground_path_length"   
DataType=H5T_NATIVE_FLOAT   
DimList=("YDim","XDim")   
MaxdimList=("YDim","XDim")   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4

END_OBJECT=DataField_4   
OBJECT=DataField_5

DataFieldName="sun_zenith"   
DataType=H5T_NATIVE_FLOAT   
DimList=("YDim","XDim")   
MaxdimList=("YDim","XDim")   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4

END_OBJECT=DataField_5   
OBJECT=DataField_6

DataFieldName="sun_azimuth"   
DataType=H5T_NATIVE_FLOAT   
DimList=("YDim","XDim")

PLANET.COM

22  
![background image](images/Planet-Tanager-UserDocumentation023.png)

Tanager Product Speciﬁcation

November 2025

MaxdimList=("YDim","XDim")   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4

END_OBJECT=DataField_6   
OBJECT=DataField_7

DataFieldName="beta_cloud_mask"   
DataType=H5T_NATIVE_UINT   
DimList=("YDim","XDim")   
MaxdimList=("YDim","XDim")   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4

END_OBJECT=DataField_7   
OBJECT=DataField_8

DataFieldName="beta_cirrus_mask"   
DataType=H5T_NATIVE_UINT   
DimList=("YDim","XDim")   
MaxdimList=("YDim","XDim")   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4

END_OBJECT=DataField_8   
OBJECT=DataField_9

DataFieldName="nodata_pixels"   
DataType=H5T_NATIVE_UINT   
DimList=("YDim","XDim")   
MaxdimList=("YDim","XDim")   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4

END_OBJECT=DataField_9   
OBJECT=DataField_10

DataFieldName="time"   
DataType=H5T_NATIVE_DOUBLE   
DimList=("YDim","XDim")   
MaxdimList=("YDim","XDim")   
CompressionType=HE5_HDFE_COMP_DEFLATE   
DeflateLevel=4

END_OBJECT=DataField_10

END_GROUP=DataField

END_GROUP=GRID_1

END_GROUP=GridStructure   
GROUP=PointStructure   
END_GROUP=PointStructure   
GROUP=ZaStructure   
END_GROUP=ZaStructure   
END

PLANET.COM

23  
![background image](images/Planet-Tanager-UserDocumentation024.png)

Tanager Product Speciﬁcation

November 2025

4. TANAGER DERIVED PRODUCTS

4.1 TANAGER METHANE NAMING CONVENTION

The name of each detected and quantiﬁed methane plume from Tanager imagery is designed to be   
unique and allow for easier recognition and sorting of the imagery. It includes the date and time of   
capture, as well as the id of the satellite that captured it providing a unique plume ID for each. The   
name of each downloaded image product is composed of the following elements:

TanagerMethane

:

\<item_id

*\>_*

\<asset_type\> .\<extension\>

Example: 20250815_043916_87_4001_ortho_ql_ch4.tif   
Searchable product in Data API is:

TanagerMethane

20250815_043916_87_4001

4.2 TANAGER METHANE PRODUCT SPECIFICATION

Planet delivers

TanagerMethane

using the methods described in the

[Carbon Mapper L3/L4 Algorithm](https://assets.carbonmapper.org/documents/L3_L4%20Algorithm%20Theoretical%20Basis%20Document_formatted_10-24-24.pdf)

[Theoretical Basis Document](https://assets.carbonmapper.org/documents/L3_L4%20Algorithm%20Theoretical%20Basis%20Document_formatted_10-24-24.pdf)

, including peer-reviewed detection, quantiﬁcation, and quality-control

approaches to ensure scientiﬁc rigor across all methane products. Each

TanagerMethane

item includes

plumes that a trained analyst can identify within corresponding

TanagerScenes

, derived from one or

more L2b methane concentration maps.

PLANET.COM

24  
![background image](images/Planet-Tanager-UserDocumentation025.png)

Tanager Product Speciﬁcation

November 2025

Planet continuously maintains and updates its methane detection products after the initial acquisition   
to incorporate any changes identiﬁed through ongoing quality control. These updates ensure that   
detections remain as accurate and authoritative as possible.

Additions, deletions, or other modiﬁcations to plumes are reﬂected in the

TanagerMethane updated

metadata ﬁeld in Planet's Data API. When this ﬁeld changes to a new date, customers should trigger a   
re-download using the Orders API to obtain the latest version Re-downloading previously consumed   
items does not affect quota, as only the initial download is counted.

A visual color ramp will be used that represents the detected concentration of methane within   
identiﬁed plumes, in ppm-m will be in an 8-bit GeoTIFF in order to communicate the spatial distribution   
of each plume. This representation of methane plumes are designed to illustrate the rough size of a   
potential emission.

Table 4-A: Tanager Derived Product Asset-Types

Item-Type

Asset-Type

Description

**TanagerMethane**

Ortho Methane QuickLook   
Plume

ortho_ql_ch4

Preliminary 8-bit scaled plume intensity in (ppm-m)   
parts-per-million-meter, in GeoTIFF format. The image will   
contain an alpha channel indicating pixels with no plume   
detections. Represents plumes detected within the initial   
72 hours after image acquisition.

Methane QuickLook Plume   
Metadata

ql_ch4_json

Preliminary plume locations, length, size in kg/hr and   
conﬁdence measure indicating the level of interpretation   
certainty, in GeoJSON format. Represents plumes detected   
within the initial 72 hours after image acquisition.

Recent Monthly Mosaic

recent_monthly_mosaic

RGB contextual baselayer from the most recent   
PlanetScope Global Monthly Mosaic, in GeoTIFF format.

4.2.1 Methane QuickLook (MQL) Product

Tanager's Methane QuickLook is a low latency, derived methane product that will reveal point source   
emission locations and estimates of magnitude in kg/hr for observed plumes This product will be   
delivered to customers within a 72 hour timeframe after acquisition enabling customers to quickly   
identify and take action where emissions are detected. Review

[Table 6-B](Planet-Tanager-UserDocumentation.html#30)

for the metadata provided

with each product in the

ql_ch4_json

asset.

Table 4-B: Tanager Methane QuickLook Assets

Asset Name

Asset-Type

Description

Ortho Methane QuickLook Plume

ortho_ql_ch4

Preliminary 8-bit scaled methane   
plume intensity in (ppm-m)   
parts-per-million-meter, in GeoTIFF   
format. The image will contain an alpha

PLANET.COM

25  
![background image](images/Planet-Tanager-UserDocumentation026.png)

Tanager Product Speciﬁcation

November 2025

channel indicating pixels with no   
plume detections.

MethaneQuickLook Plume Metadata

ql_ch4_json

Preliminary plume locations, length,   
size in kg/hr and conﬁdence measure   
indicating the level of visual   
interpretation certainty, in GeoJSON   
format.

5. PRODUCT PROCESSING

5.1 TANAGER PROCESSING

Several processing steps are applied to Tanager imagery products, listed in the table below.

Table 5-A: Tanager Processing Steps

TANAGER PROCESSING STEPS

Step

Description

Dark Subtraction

Corrects for sensor bias and dark level to ensure that zero illumination corresponds   
to zero radiance. The correction is updated frequently by averaging dark frames   
acquired over the non-sunlit side of Earth.

Pedestal Correction

Subtracts remaining residual error in the zero point after the dark frame is   
subtracted so that the numerical zero is equivalent to the radiometric zero. This   
residual is estimated by computing the median value of masked pixels located at   
the edges of the detector, which are physically blocked from external illumination.

Flat Field Correction

Corrects relative differences in pixel sensitivities to match those in the optimal   
response area of the sensor. Flat ﬁelds are collected for each optical instrument in   
lab conditions prior to launch, and are routinely updated on-orbit during the   
satellite lifetime.

Bad Pixel Correction

Fills in defective pixels on the detector following the method described in

[Chapman](https://www.mdpi.com/2072-4292/11/18/2129)

[et al. (2019)](https://www.mdpi.com/2072-4292/11/18/2129)

[,](https://www.mdpi.com/2072-4292/11/18/2129) which replaces pixels by linearly interpolating to the most similar

spectrum within the frame.

Optical Scatter Correction

Removes stray light artifacts from scatter in the optical elements to bring the   
spectral response function (SRF) towards a Gaussian distribution. These artifacts are   
modeled as concentric Gaussians convolved with the original spectrum, so   
correction involves deconvolving the stray response components from the   
spectrum with a method outlined in

[Thompson et al. (2018a)](https://www.sciencedirect.com/science/article/abs/pii/S0034425717304261)

[.](https://www.sciencedirect.com/science/article/abs/pii/S0034425717304261)

Optical Ghost Correction

Follows the correction approach of Zandbergen et al. (2020) to remove structured   
stray light artifacts ("ghosts") that arise due to unwanted reﬂections within the

PLANET.COM

26  
![background image](images/Planet-Tanager-UserDocumentation027.png)

Tanager Product Speciﬁcation

November 2025

optics. A ghost image is predicted for each frame and subsequently subtracted to   
remove the stray signal.

Absolute Radiometric Calibration

Converts the observations from Digital Number (DN) values into physical radiance   
units (W/(m²\*sr\*μm)).

Order Sorting Filter (OSF) Seam   
Correction

Interpolates over the radiometrically suspect rows where the order sorting ﬁlter   
(OSF) seams are located.

Visual Product Processing

Presents the imagery as natural color, as seen by the human eye. Only applied to   
the ortho_visual asset type.

Orthorectiﬁcation

The orthorectiﬁcation process is a method to correct the geographic location of   
imagery. The orthorectiﬁcation process depends on the accuracy of the reference   
imagery, the terrain model, satellite and sensor parameters.

OneAtlas Airbus imagery is used as reference images during Tanager   
orthorectiﬁcation and the terrain model used for the orthorectiﬁcation process is   
derived from multiple sources (SRTM, Intermap, and other local elevation datasets)   
which are periodically updated.

The orthorectiﬁcation process consists of two key steps. The ﬁrst step is a   
feature-based approach for coarse model reﬁnement followed by area-based   
matching for ﬁne model reﬁnement. The algorithm provides an improved sensor   
model of the satellite state and sensor, allowing for more accurate georectiﬁcation.

Atmospheric Correction

Removes atmospheric effects and estimates surface reﬂectance. Per pixel surface   
reﬂectance values are calculated using the ISOFIT (Imaging Spectrometer Optimal   
FITting) python package. This uses an optimal estimation method for   
simultaneously solving for both the atmospheric composition and surface   
reﬂectance values using hyperspectral radiance imagery as the input.

5.2 RADIOMETRIC INTERPRETATION

Radiance products are observed as top of atmosphere radiance. Prior to launch, the instrument   
calibration uncertainty is required to be ≤ to 15%. After launch, instrument calibration accuracy and   
uncertainty will be monitored using vicarious collections of Radiometric Calibration Network   
(RadCalNet) and other calibration sites.

All Tanager satellite images are collected at a bit depth of 16 bits and stored on-board the satellites with   
a bit depth of up to 16 bits. Radiometric corrections are applied during ground processing and all   
radiance images are delivered in 32-bit ﬂoating point precision with a unit of W/(m²\*sr\*μm).

5.2.1 Radiance Products

Tanager radiance products are calibrated hyperspectral imagery products that have been processed to   
enable analysts to derive information products for data science and analytics. The radiance product is   
optimal for value-added image processing such as land cover classiﬁcations. The imagery has

PLANET.COM

27  
![background image](images/Planet-Tanager-UserDocumentation028.png)

Tanager Product Speciﬁcation

November 2025

radiometric corrections applied to correct for any sensor artifacts and transformation to at-sensor   
radiance.

The resulting value is the at sensor radiance of that pixel in watts per steradian per square meter per   
micron (W/m²\*sr\*μm).

5.2.2 Surface Reﬂectance Products

Tanager's surface reﬂectance asset corrects for the effects of the Earth's atmosphere, accounting for the   
molecular composition and variation with altitude along with aerosol content.

**Atmospheric Correction**

Per pixel surface reﬂectance values are calculated using the ISOFIT (Imaging Spectrometer Optimal   
FITting) python package

5

. This uses an optimal estimation method for simultaneously solving for both

the atmospheric composition and surface reﬂectance values using hyperspectral radiance imagery as   
the input

6

.

6. PRODUCT METADATA

6.1 TANAGER ITEM LEVEL METADATA

The table below describes the metadata schema for

TanagerScene

\&

TanagerMethane

items:

Table 6-A: Tanager Metadata Schema

Parameter

Description

Type

_permissions

Assets available for the item which the   
authenticated user has permission to   
download.

array

geometry

Geographic boundary of the item's   
footprint, formatted as a GeoJSON   
polygon.

json

id

Globally unique item identiﬁer

string

acquired

The RFC 3339 acquisition time of the   
image.

datetime

6

Thompson, David R., Natraj, Vijay, Green, Robert O., Helmlinger, Mark C., Gao, Bo-Cai, \& Eastwood, Michael L. (2018). Optimal

estimation for imaging spectrometer atmospheric correction. Remote Sensing of Environment 216, 355-373.

5

https://github.com/isoﬁt/isoﬁt

PLANET.COM

28  
![background image](images/Planet-Tanager-UserDocumentation029.png)

Tanager Product Speciﬁcation

November 2025

cloud_percent

Percent of cloud values in the dataset.   
Cloud values represent scene content   
areas (non-blackﬁlled) that contain   
opaque clouds which prevent reliable   
interpretation of the land cover content.

double

collection_mode

Maximum Sensitivity (4x8), Standard   
Sensitivity (1x8), Glint

string

ground_control

If the image meets the positional   
accuracy speciﬁcations this value will be   
true. If the image has uncertain   
positional accuracy, this value will be   
false.

boolean

gsd

The round sampling distance of the   
associated Item. Computed for each   
item to at the center of the scene.

double

item_type

The name of the item type. For example,   
TanagerScene or TanagerMethane.

string (e.g. TanagerScene)

light_haze_percent

Fraction of the scene affected by high   
altitude cirrus clouds. The clouds are   
detected by thresholding Tanager's 1.38   
micron cirrus band. False positive   
detections are common in   
high-elevation regions.

double

pixel_resolution

Pixel resolution of the ortho products   
associated with this Item.

double

provider

Name of the imagery provider.

string (e.g. "tanager","planetscope", "skysat")

published

The RFC 3339 timestamp at which this   
item was added to the API.

datetime

publishing_stage

Stage of publishing for an item.   
TanagerScene items will be ﬁrst   
published and remain in "ﬁnalized"   
stage.

string

quality_category

Metric for image quality. To qualify for   
"standard" image quality an image must   
meet a variety of quality standards. If the   
image does not meet these criteria it is   
considered "test" quality.

string: "standard" or "test"

PLANET.COM

29  
![background image](images/Planet-Tanager-UserDocumentation030.png)

Tanager Product Speciﬁcation

November 2025

satellite_azimuth

Tanager's off track pointing direction, in   
degrees (0-360)

at the center of the

scene.

double

satellite_id

Globally unique identiﬁer of the satellite   
that acquired the underlying imagery.   
I.E. 4001

string

strip_id

The unique identiﬁer of the image strip   
that the item came from.

string

sun_azimuth

The angle of the sun, as seen by the   
observer, measured clockwise from the   
north (0 - 360) at the center of the   
scene.

double

sun_elevation

The angle of the sun above the horizon   
(0 - 90) at the center of the scene.

double

updated

The RFC 3339 timestamp at which this   
item was updated in the API.

datetime

view_angle

The satellite's off-nadir viewing angle at   
the center of the scene.

double

6.2 TANAGER DERIVED PRODUCTS

6.2.1 Methane QuickLook

The table below describes the GeoJSON metadata schema for TanagerMethane Methane QuickLook   
(MQL)

ql_ch4_json

assets:

Table 6-B: Tanager Methane Plume Metadata Schema

Parameter

Description

Type

plume_id

Unique identiﬁer for each plume. The format is   
\<strip_id_\<part\>.

The "part" postﬁx (e.g., "A", "B", "C") identiﬁes multiple   
plumes captured in the same image in the order they   
were detected.

string

plume_provider

Identiﬁes the organization who identiﬁed the   
plume.

string

plume_provider_id

Plume ID according to the

plume_provider

.

string

PLANET.COM

30  
![background image](images/Planet-Tanager-UserDocumentation031.png)

Tanager Product Speciﬁcation

November 2025

Planet assigns a new plume ID to match existing   
naming conventions.

plume_provider_version

Plume provenance information. This information   
is not directly useful to customers, but can be   
used by Planet and the

plume_provider

to

inspect data quality issues.

string

plume_quality

\*\* This ﬁeld is only used for Methane   
QuickLook,

ql_ch4_json

asset-type. \*\*

Qualitative assessment of plume quality   
captured by a human operator during the   
plume detection process (good, questionable   
or bad). More on Plume quality classiﬁcation   
below.

string

datetime

Date and time of the acquisition in   
Coordinated Universal Time (UTC)

datetime

ime

The total kilograms (kg) of methane in a plume above   
the background concentration at the time of image   
capture

ﬂoat

fetch

Plume length, meters (m)

ﬂoat

emission

Quantiﬁed emission rate of a plume in kg/hr,   
estimated using the Integrated Methane   
Enhancement method (Duren et al., 2019 -   
"California's Methane Super-Emitters", Nature)

emission_uncertainty

The uncertainty in an emission rate, derived from   
uncertainty in IME and wind speed

wind_speed_avg

Mean wind speed m/s

ﬂoat

wind_speed_std

Standard deviation wind speed m/s

ﬂoat

wind_direction_avg

Wind direction (degrees)

ﬂoat

wind_direction_std

Wind direction standard deviation (degrees)

ﬂoat

wind_source

Wind source from reanalysis (e.g. Openmeteo, HRRR,   
ERA5)

string

strip_id

Strip ID of the Tanager collect that was used for   
plume detection

string

**Plume Quality Classiﬁcation:**

Planet classiﬁes detected plumes into three categories:

**Good, Questionable**

and

**Bad**

, based on the

following criteria:

●

**Good**

: The plume is unambiguous, with a well-deﬁned shape and minimal artifacts, making it

reliable for further analysis.

●

**Questionable**

: The plume is clearly present, but there are issues. For example: irregular shape or

retrieval artifacts that could affect the accuracy of quantiﬁcation. A post-emission QC process

PLANET.COM

31  
![background image](images/Planet-Tanager-UserDocumentation032.png)

Tanager Product Speciﬁcation

November 2025

will also assess whether

**Questionable**

plumes are suitable for publishing an emission rate or if

only the detection will be reported.

●

**Bad**

: It is unclear whether the detected feature is a plume. However, this is not a ﬁnal state. After

an initial review, a secondary post-emission quality control process is conducted. This review   
determines whether the detection should be discarded or reclassiﬁed as "Questionable."

APPENDIX A -- IMAGE SUPPORT DATA

All Tanager imagery products are accompanied by a set of image support data (ISD) ﬁles. These ISD ﬁles   
provide important information regarding the image and are useful sources of ancillary data related to   
the image. The ISD ﬁles are:

●

Beta usable data mask ﬁle

●

Geolocation array

Each ﬁle is described along with its contents and format in the following sections.

BETA USABLE DATA MASK FILE

The Beta usable data mask information that is stored in the HDF-EOS5 Basic Products (see

section

[Beta Usable Data Mask in section 3.2](Planet-Tanager-UserDocumentation.html#18)

) can also be accessed through the standalone asset,

ortho_beta_udm

, in the ﬁle format GeoTiff. The usable data mask ﬁle provides information on areas of

usable data within an image (e.g. clear, cloud, or cirrus).

The pixel size after orthorectiﬁcation will be 30 m for Tanager ortho_beta_udm. The usable data

mask is a raster image having the same dimensions as the image product, comprising 3 bands, where   
each band represents a speciﬁc usability class mask. The usability masks are mutually exclusive, and a   
value of one indicates that the pixel is assigned to that usability class.

●

Band 1: Beta cloud mask (a value of "1" indicates the pixel has clouds, a value of "0" indicates that   
the pixel does not have clouds)

●

Band 2: Beta cirrus mask

●

Band 3: NODATA pixel mask

GEOLOCATION ARRAY

The geolocation information that is stored in the HDF-EOS5 Basic Products can also be accessed   
through the standalone asset,

geolocation_array

, in the ﬁle format GeoTiff. The geolocation arrays have

the same dimensions as the basic imagery products (e.g. Basic Radiance Scene or Basic Surface   
Reﬂectance Scene) with 2 bands. See also[](Planet-Tanager-UserDocumentation.html#20)

[Geolocation Fields in Section 3.2](Planet-Tanager-UserDocumentation.html#20)

. The metadata of the GeoTiff

also contains the

*Planet_Ortho_Framing*

information explained in Section 3.2.

PLANET.COM

32  
![background image](images/Planet-Tanager-UserDocumentation033.png)

Tanager Product Speciﬁcation

November 2025

●

Band 1: Longitude in Decimal Degrees

●

Band 2: Latitude in Decimal Degrees

PLANET.COM

33
