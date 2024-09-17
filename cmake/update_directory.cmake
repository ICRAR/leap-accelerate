file(GLOB_RECURSE _file_list RELATIVE "${src_dir}" "${src_dir}/*")
 
foreach(each_file ${_file_list})
  set(destinationfile "${dst_dir}/${each_file}")
  set(sourcefile "${src_dir}/${each_file}")
  
  file(TIMESTAMP ${sourcefile} SRC_TIMESTAMP)
  file(TIMESTAMP ${destinationfile} DST_TIMESTAMP)
  
  if(NOT EXISTS ${destinationfile})
    set(DST_TIMESTAMP 0)
  endif()
  if(NOT EXISTS ${destinationfile} OR ${SRC_TIMESTAMP} STRGREATER ${DST_TIMESTAMP})
    get_filename_component(destinationdir ${destinationfile} DIRECTORY)
    file(COPY ${sourcefile} DESTINATION ${destinationdir})
    message(STATUS "copying ${sourcefile} -> ${destinationdir}")
  endif()
endforeach(each_file)