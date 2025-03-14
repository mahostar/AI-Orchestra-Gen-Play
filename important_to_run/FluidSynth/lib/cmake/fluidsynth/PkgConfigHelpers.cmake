macro ( sanitize_property_dirs target property )
        set(_cleandirs)
        get_target_property(_dirs ${target} ${property})
        if(_dirs)
            foreach(_d IN LISTS _dirs)
                if(EXISTS ${_d})
                    list(APPEND _cleandirs ${_d})
                else()
                    message(DEBUG "removing spurious directory ${_d} from property ${property} of target ${target}")
                endif()
            endforeach()
            set_property(TARGET ${target} PROPERTY ${property} ${_cleandirs})
        endif()
endmacro ( sanitize_property_dirs )

macro ( sanitize_target_dirs target )
    if (TARGET ${target})
        message(DEBUG "performing sanitize_target_dirs(${target})")
        sanitize_property_dirs( ${target} INTERFACE_INCLUDE_DIRECTORIES )
        sanitize_property_dirs( ${target} INTERFACE_SYSTEM_INCLUDE_DIRECTORIES )
        sanitize_property_dirs( ${target} INTERFACE_LINK_DIRECTORIES )
    endif()
endmacro ( sanitize_target_dirs )

macro ( generate_pkgconfig_spec template outfile target )
    #message(DEBUG "generate_pkgconfig_spec: ${outfile} from template: ${template}")
    if (TARGET ${target})
        # retrieve all the private libs we depend on
        get_target_property (_libs ${target} INTERFACE_LINK_LIBRARIES)
        set(_cleanlibs)
        foreach(_lib IN LISTS _libs)
            if (TARGET ${_lib})
                # All the imported PkgConfig target are explicitly added to PC_REQUIRES_PRIV.
                # Do not duplicate them into the Libs.private section, as they will be already part of Requires.private
            else()
                list(APPEND _cleanlibs ${_lib})
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _cleanlibs)
        list ( REMOVE_DUPLICATES PC_LIBS_PRIV )
        set (LIBS_PRIVATE ${_cleanlibs} ${PC_LIBS_PRIV})
        # make a copy
        set ( LIBS_PRIVATE_WITH_PATH ${LIBS_PRIVATE} )

        # this matches any path and any flag entries (starting with '-')
        set ( LIB_LIST_REGEX "(^(.+)\/([^\/]+)$)|(^\-.*$)" )
        # remove all entries from the list which are specified by path and which already have -l
        list ( FILTER LIBS_PRIVATE EXCLUDE REGEX ${LIB_LIST_REGEX} )
        # include only entries specified by path
        list ( FILTER LIBS_PRIVATE_WITH_PATH INCLUDE REGEX ${LIB_LIST_REGEX} )

        # prepend the linker flag to all entries except the ones that already have it
        list ( TRANSFORM LIBS_PRIVATE PREPEND "-l")
        list ( JOIN LIBS_PRIVATE " " LIBS_PRIVATE_JOINED )
        list ( JOIN LIBS_PRIVATE_WITH_PATH " " LIBS_PRIVATE_WITH_PATH_JOINED )
        
        list ( JOIN PC_REQUIRES_PRIV " " PC_REQUIRES_PRIV_JOINED )

        configure_file ( ${template} ${outfile} IMMEDIATE @ONLY)
    endif()
endmacro ( generate_pkgconfig_spec )

macro ( unset_pkg_config _prefix )
  unset ( ${_prefix}_VERSION CACHE )
  unset ( ${_prefix}_PREFIX CACHE )
  unset ( ${_prefix}_CFLAGS CACHE )
  unset ( ${_prefix}_CFLAGS_OTHER CACHE )
  unset ( ${_prefix}_LDFLAGS CACHE )
  unset ( ${_prefix}_LDFLAGS_OTHER CACHE )
  unset ( ${_prefix}_LIBRARIES CACHE )
  unset ( ${_prefix}_INCLUDEDIR CACHE )
  unset ( ${_prefix}_INCLUDE_DIRS CACHE )
  unset ( ${_prefix}_LIBDIR CACHE )
  unset ( ${_prefix}_LIBRARY_DIRS CACHE )
  unset ( __pkg_config_checked_${_prefix} CACHE )
endmacro ( unset_pkg_config )

function ( get_target_properties_from_pkg_config _library _prefix _out_prefix )
  if ( NOT "${_library}" MATCHES "${CMAKE_IMPORT_LIBRARY_SUFFIX}$"
       AND "${_library}" MATCHES "${CMAKE_STATIC_LIBRARY_SUFFIX}$" )
    set ( _cflags ${_prefix}_STATIC_CFLAGS_OTHER )
    set ( _link_libraries ${_prefix}_STATIC_LIBRARIES )
    set ( _library_dirs ${_prefix}_STATIC_LIBRARY_DIRS )
  else ()
    set ( _cflags ${_prefix}_CFLAGS_OTHER )
    set ( _link_libraries ${_prefix}_LIBRARIES )
    set ( _library_dirs ${_prefix}_LIBRARY_DIRS )
  endif ()

  # The link_libraries list always starts with the library itself, and POP_FRONT is >=3.15
  list(REMOVE_AT "${_link_libraries}" 0)

  set ( ${_out_prefix}_compile_options "${${_cflags}}" PARENT_SCOPE )
  set ( ${_out_prefix}_link_libraries "${${_link_libraries}}" PARENT_SCOPE )
  set ( ${_out_prefix}_link_directories "${${_library_dirs}}" PARENT_SCOPE )
endfunction ( get_target_properties_from_pkg_config )
