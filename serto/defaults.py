class Defaults:
    """
    Default constants for the SERTO package
    """
    # Default number of threads to use for each SWMM simulation
    THREADS_PER_SIMULATION: int = 4

    # Precipitation frequency data server url
    NOAA_PFDS_REST_SERVER_URL: str = r'https://hdsc.nws.noaa.gov/cgi-bin/hdsc/new/'
