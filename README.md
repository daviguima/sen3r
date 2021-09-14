## SEN3R - Sentinel 3 Reflectance Retrieval over Rivers

SEN3R is a stand-alone command-line utility made to simplify the pipeline of image 
processing over ESA's Sentinel-3 mission without the hassle of installing [SNAP](https://step.esa.int/main/toolboxes/snap/) as a dependency 
(We have no intention of replacing SNAP). 
<br>
<br>
[!] WARNING: GDAL is a requirement for the installation therefore, 
usage of a conda environment 
([Anaconda.org](https://www.anaconda.com/products/individual)) 
is strongly recommended. Unless you know what you are doing (-:
<br>
<br>
Create a Conda environment (not all python versions above 3.7 were tested but they should also be compatible):
```
conda create --name sen3r python=3.7
```
Activate your conda env:
```
conda activate sen3r
```
Install GDAL before installing `requirements.txt` to avoid dependecy error with pyshp:
```
conda install -c conda-forge gdal
```
Install the requirements:
```
pip install -r requirements.txt
```
Run the setup:
```
python setup.py install 
```
Do a quick test:
```
sen3r -h 
```
If all runs well, you should see:
```
(sen3r) C:\yourpath\sen3r>sen3r -h
usage: sen3r [-h] [-i INPUT] [-o OUT] [-r ROI] [-p PRODUCT] [-c CAMS] [-ng] [-np] [-s] [-v] [-l]

SEN3R (Sentinel-3 Reflectance Retrieval over Rivers) enables extraction of reflectance time series from Sentinel-3 L2 WFR images over water bodies.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        The products input folder. Required.
  -o OUT, --out OUT     Output directory. Required.
  -r ROI, --roi ROI     Region of interest (SHP, KML or GeoJSON). Required
  -p PRODUCT, --product PRODUCT
                        Currently only WFR is available.
  -c CAMS, --cams CAMS  Path to search for auxiliary CAMS file. Optional.
  -ng, --no-graphics    Do not generate graphical reports.
  -np, --no-pdf         Do not generate PDF report.
  -s, --single          Single mode: run SEN3R over only one image instead of a whole directory.Optional.
  -v, --version         Displays current package version.
  -l, --silent          Run silently, stop printing to console.

```
Usage:
```
sen3r -i "C:\PATH\TO\L2_WFR_FILES" -o "C:\sen3r_out" -r "C:\path\to\your_vector.kml"
```
Currently supported vector formats for `-r` are: `.shp`, `.kml`, `.kmz`, `.json` and `.geojson`
