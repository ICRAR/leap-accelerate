#!/bin/bash
#
# ICRAR - International Centre for Radio Astronomy Research
# (c) UWA - The University of Western Australia, 2018
# Copyright by UWA (in the framework of the ICRAR)
# All rights reserved
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston,
# MA 02111-1307  USA
#

<<<<<<< HEAD
wget "https://cloudstor.aarnet.edu.au/plus/s/YoYdODmk9iVS5Sq/download" -O ./mwa/1197638568-32.tar.gz
tar -C ./ -xvf ./mwa/1197638568-32.tar.gz

=======
mkdir mwa
>>>>>>> 7c57dd8... testdata mkdir
wget "https://cloudstor.aarnet.edu.au/plus/s/Eb65Nqy66hUE2tO/download" -O ./mwa/1197638568-split.tar.gz
tar -C ./ -xvf ./mwa/1197638568-split.tar.gz

mkdir ska
wget "https://cloudstor.aarnet.edu.au/plus/s/qtIV1HqXfKsQVAu/download" -O ./ska/SKA_LOW_SIM_short_EoR0_ionosphere_off_GLEAM.0001.tar.gz
tar -C ./ -xvf ./ska/SKA_LOW_SIM_short_EoR0_ionosphere_off_GLEAM.0001.tar.gz