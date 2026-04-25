

::: mermaid
mindmap
  root((Spectral Complexity))
    Enhancements
        Sliding Window
            Num Encmembers Extracted
                id)2x2 window with 3-4 endmembers(
            Window Size
                id)plot volume increasing from small to frame dim(
                non-square windows
            Skip averaging of valid windows
            Pixel mask before calculation
                id))invalid pixels currently poisoning neighborhood volumes((
        Endmember Selection
            Gantmacher h search
            NfindR
        Registration Methods
            Albers Equal Area
            Fourier spatial match
        Volume Estimation
            QR decomposition
            max of Gram volume function
                id))volume dimensionality mismatch((
                quantify # endmembers most often result in max
    Use Cases
        Construction detection
        Ship wake detection
        Natural Disaster Mapping
            Find canonical example and test
            Landslide
            Forest fire
            Chemical spills
            Tephra fallout
    Theory Questions
        Connection to manifolds?
        Correlate pixel volume to manifold snake?
:::