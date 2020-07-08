# +-----------------------------------------------------------------------------+
# |   Copyright (C) 2007-2011                                                   |
# |   Lars B"ahren (lbaehren@gmail.com)                                         |
# |                                                                             |
# |   This program is free software; you can redistribute it and/or modify      |
# |   it under the terms of the GNU General Public License as published by      |
# |   the Free Software Foundation; either version 2 of the License, or         |
# |   (at your option) any later version.                                       |
# |                                                                             |
# |   This program is distributed in the hope that it will be useful,           |
# |   but WITHOUT ANY WARRANTY; without even the implied warranty of            |
# |   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the             |
# |   GNU General Public License for more details.                              |
# |                                                                             |
# |   You should have received a copy of the GNU General Public License         |
# |   along with this program; if not, write to the                             |
# |   Free Software Foundation, Inc.,                                           |
# |   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.                 |
# +-----------------------------------------------------------------------------+

# - Check for the presence of CASACORE
#
# The following variables are set when CASACORE is found:
#  CASACORE_FOUND      = Set to true, if all components of CASACORE have been found.
#  CASACORE_INCLUDES   = Include path for the header files of CASACORE
#  CASACORE_LIBRARIES  = Link these to use CASACORE

if (NOT CASACORE_FOUND)

  if (NOT CASACORE_ROOT_DIR)
    set (CASACORE_ROOT_DIR ${CMAKE_INSTALL_PREFIX})
  endif (NOT CASACORE_ROOT_DIR)
  
  set (CASACORE_MODULES
    casa
    tables
    mirlib
    scimath_f
    scimath
    measures
    fits
    coordinates
    components
    lattices
    ms
    images
    msfits
    )

  set (CASACORE_HEADERS
    casa/Arrays.h
    tables/Tables.h
    mirlib/miriad.h
    scimath/Fitting.h
    measures/Measures.h
    fits/FITS.h
    coordinates/Coordinates.h
    components/ComponentModels.h
    lattices/Lattices.h
    ms/MeasurementSets.h
    images/Images.h
    msfits/MSFits.h
    )
  
  ##_____________________________________________________________________________
  ## Check for the header files
  
  find_path (CASACORE_INCLUDES ${CASACORE_HEADERS}
    HINTS ${CASACORE_ROOT_DIR}
    PATH_SUFFIXES include include/casacore
    )

  ##_____________________________________________________________________________
  ## Check for the library
  
  set (CASACORE_LIBRARIES "")

  foreach (_casacore_lib ${CASACORE_MODULES})
    
    ## Convert library name to CMake variable
    string (TOUPPER ${_casacore_lib} _casacore_var)

    ## Search for the library
    find_library (CASACORE_${_casacore_var}_LIBRARY
      NAMES casa_${_casacore_lib} ${_casacore_lib}
      HINTS ${CASACORE_ROOT_DIR}
      PATH_SUFFIXES lib
      )
    
    ## If library was found, add it to the list of libraries
    if (CASACORE_${_casacore_var}_LIBRARY)
      list (APPEND CASACORE_LIBRARIES ${CASACORE_${_casacore_var}_LIBRARY})
    endif (CASACORE_${_casacore_var}_LIBRARY)

  endforeach (_casacore_lib)

  ## adjust the ordering of the individual libraries
  
  list (REVERSE CASACORE_LIBRARIES)

  ##_____________________________________________________________________________
  ## Check for measures data
  
  find_path (CASACORE_DATADIR ephemerides geodetic
    HINTS ${CASACORE_ROOT_DIR}
    PATH_SUFFIXES share/casacore share/casacore/data
    )
  
  ##_____________________________________________________________________________
  ## Actions taken when all components have been found
  
  if (CASACORE_INCLUDES AND CASACORE_LIBRARIES)
    set (CASACORE_FOUND TRUE)
  else (CASACORE_INCLUDES AND CASACORE_LIBRARIES)
    set (CASACORE_FOUND FALSE)
    if (NOT CASACORE_FIND_QUIETLY)
      if (NOT CASACORE_INCLUDES)
	message (STATUS "Unable to find CASACORE header files!")
      endif (NOT CASACORE_INCLUDES)
      if (NOT CASACORE_LIBRARIES)
	message (STATUS "Unable to find CASACORE library files!")
      endif (NOT CASACORE_LIBRARIES)
    endif (NOT CASACORE_FIND_QUIETLY)
  endif (CASACORE_INCLUDES AND CASACORE_LIBRARIES)
  
  if (CASACORE_FOUND)
    if (NOT CASACORE_FIND_QUIETLY)
      message (STATUS "Found components for CASACORE")
      message (STATUS "CASACORE_INCLUDES       = ${CASACORE_INCLUDES}")
      message (STATUS "CASACORE_LIBRARIES      = ${CASACORE_LIBRARIES}")
      message (STATUS "... libcasa_casa        = ${CASACORE_CASA_LIBRARY}")
      message (STATUS "... libcasa_tables      = ${CASACORE_TABLES_LIBRARY}")
      message (STATUS "... libcasa_scimath     = ${CASACORE_SCIMATH_LIBRARY}")
      message (STATUS "... libcasa_measures    = ${CASACORE_MEASURES_LIBRARY}")
      message (STATUS "... libcasa_coordinates = ${CASACORE_COORDINATES_LIBRARY}")
      message (STATUS "... libcasa_ms          = ${CASACORE_MS_LIBRARY}")
      message (STATUS "... libcasa_images      = ${CASACORE_IMAGES_LIBRARY}")
    endif (NOT CASACORE_FIND_QUIETLY)
  else (CASACORE_FOUND)
    if (CASACORE_FIND_REQUIRED)
      message (FATAL_ERROR "Could not find CASACORE!")
    endif (CASACORE_FIND_REQUIRED)
  endif (CASACORE_FOUND)

  ## Compatibility variables

  set (CASA_FOUND       ${CASACORE_FOUND}     )
  set (CASA_INCLUDES    ${CASACORE_INCLUDES}  )
  set (CASA_LIBRARIES   ${CASACORE_LIBRARIES} )
  
  ##_____________________________________________________________________________
  ## Mark advanced variables
  
  mark_as_advanced (
    CASACORE_ROOT_DIR
    CASACORE_INCLUDES
    CASACORE_LIBRARIES
    CASACORE_CASA_LIBRARY
    CASACORE_TABLES_LIBRARY
    CASACORE_DATADIR
    )
  
endif (NOT CASACORE_FOUND)