cmake_minimum_required(VERSION 3.13.4)

project(FastParcellation)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/FastParcellation")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Aaron Kujawa (King's College London)")
set(EXTENSION_DESCRIPTION "This extension allows to run a docker container, send image(s) and a path of a  trained model to it, and obtain a segmentation of input image(s) as provided by the container.")
set(EXTENSION_ICONURL "http://www.example.com/Slicer/Extensions/FastParcellation.png")
set(EXTENSION_SCREENSHOTURLS "http://www.example.com/Slicer/Extensions/FastParcellation/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(run_command)
add_subdirectory(Full_Brain_Parcellation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
