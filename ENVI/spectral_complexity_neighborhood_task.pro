PRO spectral_complexity_neighborhood_task, $
  INPUT_RASTER=input_raster, $
  TILE_SIZE=tile_size, $
  STRIDE=stride, $
  NUM_ENDMEMBERS=num_endmembers, $
  OUTPUT_RASTER_URI=output_raster_uri, $
  OUTPUT_RASTER=output_raster, $
  _EXTRA=extra

  COMPILE_OPT idl2

  ; Set defaults
  IF (~ISA(tile_size)) THEN tile_size = 5
  IF (~ISA(stride)) THEN stride = 1
  IF (~ISA(num_endmembers)) THEN num_endmembers = 3

  e = ENVI(/CURRENT)
  
  ; Ensure output URI exists
  IF (~ISA(output_raster_uri)) THEN output_raster_uri = e.GetTemporaryFilename('dat')
  
  ; Export input raster to disk if it's virtual
  in_uri = input_raster.URI
  IF (in_uri EQ "") THEN BEGIN
    in_uri = e.GetTemporaryFilename('dat')
    input_raster.Export, in_uri, 'ENVI'
  ENDIF

  ; Call Python as a completely separate process (bypassing IDL-Python version limits)
  python_exe = 'python' ; Use system default python
  script_path = 'F:\Resilio\IMGS 890 Research\Spectral-Complexity-dev\ENVI\envi_bridge.py'
  
  cmd = python_exe + ' "' + script_path + '" neighborhood "' + in_uri + '" "' + output_raster_uri + '" ' + $
        STRTRIM(tile_size, 2) + ' ' + STRTRIM(stride, 2) + ' ' + STRTRIM(num_endmembers, 2)
  
  SPAWN, cmd, out_text, err_text
  
  ; Check for errors
  IF (N_ELEMENTS(err_text) GT 0) THEN BEGIN
    IF (err_text[0] NE "") THEN BEGIN
      FOR i=0, N_ELEMENTS(err_text)-1 DO PRINT, err_text[i]
      ; e.ReportError, 'Python Execution Error: ' + err_text[0]
    ENDIF
  ENDIF

  ; Re-open the generated raster from Python back into ENVI
  has_spatial = OBJ_VALID(input_raster.SPATIALREF)
  IF (has_spatial) THEN BEGIN
    output_raster = e.OpenRaster(output_raster_uri, SPATIALREF_OVERRIDE=input_raster.SPATIALREF)
  ENDIF ELSE BEGIN
    output_raster = e.OpenRaster(output_raster_uri)
  ENDELSE
END
