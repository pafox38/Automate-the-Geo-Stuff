# Automate the Geo Stuff
## Useful code for geoscientists / geologists in the energy sector
 * A workflow that illustrates how you can utilize public data to quickly determine what may be driving production performance for horizontal wells drilled in a particular reservoir.  In this example, we use a dataset containing over 4,000 horizontal wells targeting the Eagle Ford Shale.  Although the resulting regressor model isn't perfect, once can see that there are some key learnings extracted from the model(s).
 * las curve mnemonic aliasing.
 * las API Change.  This program will parse through a folder full of las files and standardize API numbers and output new files with the new API number as the file name.  This makes bulk loading into G&G software much easier.  Additional if/elif and Exceptions can be added if other formats are discovered.
 * Violin plot to illustrate a cross-sectional view of wellbore targeting.  Simple gun barrel plots illustrate an idealized view of post-drill wellbore targeting.  The use of violin plots provides the user to communicate WHERE the wellbore traversed in adjacent formations/units.  Utilizes data exported from geosteering software. 
 * Using machine learning (Random Forest Regressor) to create synthetic log curves.  In this example, DTC.
 * Directory file search based on file extension.  Useful when looking for pesky las files.

## Copyright and Licensing
These programs and files are free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or any later version.

© 2021 pafox38
