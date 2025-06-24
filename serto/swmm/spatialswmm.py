# python imports
from __future__ import annotations
from typing import List, Any, Union, Tuple, Dict
import re
from datetime import datetime, time

import pandas as pd
# 3rd party imports
from pyproj import CRS
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from networkx import DiGraph, MultiDiGraph


# project imports


class SpatialSWMM:
    """
    This class represents a spatial SWMM model
    """
    SECTIONS = ['[TITLE]', '[OPTIONS]', '[EVAPORATION]', '[RAINGAGES]', '[SUBCATCHMENTS]', '[SUBAREAS]',
                '[INFILTRATION]', '[JUNCTIONS]', '[OUTFALLS]', '[STORAGE]', '[CONDUITS]', '[XSECTIONS]',
                '[LOSSES]', '[POLLUTANTS]', '[LANDUSES]', '[COVERAGES]', '[LOADINGS]', '[BUILDUP]',
                '[WASHOFF]', '[TREATMENT]', '[INFLOWS]', '[DWF]', '[HYDROGRAPHS]', '[RDII]', '[UNITHYDROGRAPHS]',
                '[AQUIFERS]', '[GROUNDWATER]', '[SOURCES]', '[SOURCES]', '[MAP]', '[COORDINATES]', '[VERTICES]',
                '[POLYGONS]', '[SYMBOLS]', '[LABELS]', '[BACKDROP]', '[TAGS]', '[MAP]', '[COORDINATES]', '[VERTICES]']

    OPTIONS_COLUMNS_PARAMS = {
        "FLOW_UNITS": ["CFS", "GPM", "MGD", "CMS", "LPS", "MLD"],
        "INFILTRATION": ["HORTON", "MODIFIED_HORTON", "GREEN_AMPT", "MODIFIED_GREEN_AMPT", "CURVE_NUMBER"],
        "FLOW_ROUTING": ["KINWAVE", "DYNWAVE", "STAGWAVE"],
        "LINK_OFFSETS": ["DEPTH", "ELEVATION"],
        "MIN_SLOPE": 0,
        "ALLOW_PONDING": ["YES", "NO"],
        "SKIP_STEADY_STATE": ["YES", "NO"],

        "IGNORE_RAINFALL": ["YES", "NO"],
        "IGNORE_RDII": ["YES", "NO"],
        "IGNORE_SNOWMELT": ["YES", "NO"],
        "IGNORE_GROUNDWATER": ["YES", "NO"],
        "IGNORE_ROUTING": ["YES", "NO"],
        "IGNORE_QUALITY": ["YES", "NO"],
        "START_DATE": None,
        "START_TIME": None,
        "REPORT_START_DATE": None,
        "REPORT_START_TIME": None,
        "END_DATE": None,
        "END_TIME": None,
        "SWEEP_START": "01/01",
        "SWEEP_END": "12/31",
        "DRY_DAYS": 0,
        "REPORT_STEP": "00:05:00",
        "WET_STEP": "00:05:00",
        "DRY_STEP": "00:05:00",
        "ROUTING_STEP": "00:00:30",
        "RULE_STEP": "00:05:00",

        "INERTIAL_DAMPING": ["PARTIAL", "FULL"],
        "NORMAL_FLOW_LIMITED": ["SLOPE", "FROUDE", "BOTH"],
        "FORCE_MAIN_EQUATION": ["H-W", "D-W"],
        "SURCHARGE_METHOD": ["EXTRAN", "SLOT"],
        "VARIABLE_STEP": 0.75,
        "LENGTHENING_STEP": 0,
        "MIN_SURFAREA": 12.566,
        "MAX_TRIALS": 8,
        "HEAD_TOLERANCE": 0.001,
        "SYS_FLOW_TOL": 5,
        "LAT_FLOW_TOL": 5,
        "MINIMUM_STEP": 0.5,
        "THREADS": 1,
    }

    OPTIONS_SECTION_SPLIT = ["SKIP_STEADY_STATE", "RULE_STEP"]

    RAINGAGES_COLUMNS = ['Name', 'Format', 'Interval', 'SCF', 'Source', 'Param1', 'Param2', 'Param3', 'Param4']
    RAINGAGES_COLUMN_TYPES = ['str', 'str', 'str', 'float', 'str', 'float', 'float', 'float', 'float']
    RAINGAGES_COLUMN_LENGTHS = [31, 31, 31, 16, 31, 16, 16, 16, 16]

    COMMENTS_COLUMNS = ['Prepended_Comments']

    JUNCTION_COLUMNS = ['Elevation', 'MaxDepth', 'InitDepth', 'SurDepth', 'Aponded']
    JUNCTION_COLUMN_TYPES = ['float', 'float', 'float', 'float', 'float']
    JUNCTION_COLUMN_LENGTHS = [16, 16, 16, 16, 16]

    OUTFALL_COLUMNS = ['Elevation', 'Type', 'StageData', 'Gated', 'RouteTo']
    OUTFALL_COLUMN_LENGTHS = [16, 10, 16, 16]

    STORAGE_COLUMNS = ['Elevation', 'MaxDepth', 'InitDepth', 'Shape', 'Curve Type/Params', 'SurDepth', 'Fevap', 'Psi',
                       'Ksat', 'IMD', 'Aponded']

    SUB_CATCHMENT_COLUMNS = ['Rain Gage', 'Outlet', 'Area', '%Imperv', 'Width', '%Slope', 'CurbLength', 'SnowPack']

    SUB_AREA_COLUMNS = ['N-Imperv', 'N-Perv', 'S-Imperv', 'S-Perv', 'PctZero', 'RouteTo', 'PctRouted']

    CONDUIT_COLUMNS = ['From Node', 'To Node', 'Length', 'Roughness', 'InOffset', 'OutOffset', 'InitFlow', 'MaxFlow',
                       'Shape', 'Geom1', 'Geom2', 'Geom3', 'Geom4']

    ORIFICE_COLUMNS = ['From Node', 'To Node', 'Type', 'Offset', 'Qcoeff', 'Gated', 'CloseTime', 'OpenTime']

    WEIR_COLUMNS = ['From Node', 'To Node', 'Type', 'CrestHeight', 'Qcoeff', 'Gated', 'EndCon', 'EndCoeff', 'Surcharge',
                    'RoadWidth', 'Coeff. Curve']

    OUTLET_COLUMNS = ['From Node', 'To Node', 'Type', 'Offset', 'Type', 'QTable/Qcoeff', 'Qexpon', 'Gated']

    XSECTION_COLUMNS = ['Shape', 'Geom1', 'Geom2', 'Geom3', 'Geom4', 'Barrels', 'Culvert']

    POLLUTANTS_COLUMNS = ['Units', 'Crain', 'Cgw', 'Crdii', 'Kdecay', 'SnowOnly', 'Co-Pollutant',
                          'Co-Frac', 'Cdwf', 'Cinit']

    LANDUSES_COLUMNS = ['Sweeping Interval', 'Fraction Available', 'Last Swept']

    COVERAGES_COLUMNS = ['Land Use', 'Percent']

    LOADINGS_COLUMNS = ['Pollutant', 'Buildup', 'Washoff']

    BUILDUP_COLUMNS = ['Pollutant', 'Function', 'Coeff1', 'Coeff2', 'Coeff3', 'Per Unit']

    WASHOFF_COLUMNS = ['Pollutant', 'Function', 'Coeff1', 'Coeff2', 'SweepRemoval', 'BMPRemoval']

    INFLOW_COLUMNS = ['Node', 'Constituent', 'Time Series', 'Type', 'Mfactor', 'Sfactor', 'Baseline', 'Pattern']

    def __init__(self, inp_file: str,  crs: Union[str, CRS, int, Dict, Tuple[str, str], Any]) -> None:
        """
        This function initializes the SpatialSWMM object
        :param inp_file: The path to the SWMM input file
        :type inp_file: str
        :param crs: The coordinate reference system for the model
        Initialize a CRS class instance with:
          - PROJ string
          - Dictionary of PROJ parameters
          - PROJ keyword arguments for parameters
          - JSON string with PROJ parameters
          - CRS WKT string
          - An authority string [i.e. 'epsg:4326']
          - An EPSG integer code [i.e. 4326]
          - A tuple of ("auth_name": "auth_code") [i.e ('epsg', '4326')]
          - An object with a `to_wkt` method.
          - A :class:`pyproj.crs.CRS` class instance.
        :type crs: str, CRS, int, Dict, Tuple[str, str], Any
        """
        self._crs = crs if isinstance(crs, CRS) else CRS.from_user_input(crs)
        self._title_comments: List[str] = []
        self._title: List[str] = []

        self._options = pd.DataFrame(columns=['Option', 'Value', *SpatialSWMM.COMMENTS_COLUMNS])
        self._files = pd.DataFrame(columns=['USE/SAVE', 'Type', 'Fname'])

        # initialize tables for store
        self._nodes: gpd.GeoDataFrame = gpd.GeoDataFrame(
            columns=[],
            geometry=[],
            crs=self._crs
        )

        self._links: gpd.GeoDataFrame = gpd.GeoDataFrame(
            columns=[],
            geometry=[],
            crs=self._crs
        )

        self._sub_catchments: gpd.GeoDataFrame = gpd.GeoDataFrame(
            columns=[],
            geometry=[],
            crs=self._crs
        )

        self._raingages: gpd.GeoDataFrame = gpd.GeoDataFrame(
            columns=[],
            geometry=[],
            crs=self._crs
        )

        self._sub_areas: pd.DataFrame = pd.DataFrame(columns=SpatialSWMM.SUB_AREA_COLUMNS)
        self._xsections: pd.DataFrame = pd.DataFrame(columns=SpatialSWMM.XSECTION_COLUMNS)
        self._pollutants: pd.DataFrame = pd.DataFrame(columns=SpatialSWMM.POLLUTANTS_COLUMNS)
        self._landuses: pd.DataFrame = pd.DataFrame(columns=SpatialSWMM.LANDUSES_COLUMNS)
        self._coverages: pd.DataFrame = pd.DataFrame(columns=SpatialSWMM.COVERAGES_COLUMNS)
        self._loadings: pd.DataFrame = pd.DataFrame(columns=SpatialSWMM.LOADINGS_COLUMNS)

        self.__read_model(
            model_path=inp_file,
        )



    # def initialize(self):
    #     """
    #     Initialize the model
    #     :return:
    #     """
    #     self._title_comments = []
    #     self._title = []
    #     self._options = pd.DataFrame(columns=['Option', 'Value', *SpatialSWMM.COMMENTS_COLUMNS])
    #
    #     self.initialize_options()
    #
    # def initialize_options(self):
    #     """
    #     Initialize the options
    #     :return:
    #     """
    #
    #     start_date = datetime.now()
    #     start_date = datetime(start_date.year, start_date.month, start_date.day)
    #
    #     end_date = start_date + pd.Timedelta('1d')
    #
    #     for option_key, option_value in SpatialSWMM.OPTIONS_COLUMNS_PARAMS.items():
    #         if isinstance(option_value, list):
    #             self._options.loc[option_key] = option_value[0]
    #         elif option_key == 'START_DATE' or option_key == 'REPORT_START_DATE':
    #             self._options.loc[option_key] = start_date.strftime('%m/%d/%Y')
    #         elif option_key == 'START_TIME' or option_key == 'REPORT_START_TIME':
    #             self._options.loc[option_key] = start_date.strftime('%H:%M:%S')
    #         elif option_key == 'END_DATE':
    #             self._options.loc[option_key] = end_date.strftime('%m/%d/%Y')
    #         elif option_key == 'END_TIME':
    #             self._options.loc[option_key] = end_date.strftime('%H:%M:%S')
    #         else:
    #             self._options.loc[option_key] = option_value

    @property
    def title(self) -> List[str]:
        """
        This function returns the title of the model
        :return: The title of the model
        """
        return self._title

    @property
    def options(self):
        """:
        This function returns the options of the model
        :return: The options of the model
        """
        return self._options



    @property
    def title_comments(self) -> List[str]:
        """
        This function returns the title comments of the model
        :return: The title comments of the model
        """
        return self._title_comments

    @property
    def options(self) -> pd.DataFrame:
        """
        This function returns the options of the model
        :return: The options of the model
        """
        return self._options

    @property
    def raingages(self) -> gpd.GeoDataFrame:
        """
        This function returns the raingages of the model
        :return: The raingages of the model
        """
        return self._raingages

    @property
    def nodes(self) -> gpd.GeoDataFrame:
        """
        This function returns the junctions of the model
        :return: The junctions of the model
        """
        return self._nodes

    @property
    def subcatchments(self) -> gpd.GeoDataFrame:
        """
        This function returns the subcatchments of the model
        :return: The subcatchments of the model
        """
        return self._sub_catchments

    @property
    def sub_area(self, sub_catchment_id: str) -> pd.DataFrame:
        """
        This function returns the sub area of the subcatchment
        :param sub_catchment_id: The subcatchment id
        :return: The sub area of the subcatchment
        """
        return self._sub_areas.loc[sub_catchment_id]

    @property
    def xsection(self, link_id: str) -> pd.DataFrame:
        """
        This function returns the cross section of the link
        :param link_id: The link id
        :return: The cross section of the link
        """
        return self._xsections.loc[link_id]

    @property
    def pollutants(self) -> pd.DataFrame:
        """
        This function returns the pollutants of the model
        :return: The pollutants of the model
        """
        return self._pollutants

    @property
    def landuses(self) -> pd.DataFrame:
        """
        This function returns the landuses of the model
        :return: The landuses of the model
        """
        return self._landuses

    @property
    def coverages(self) -> pd.DataFrame:
        """
        This function returns the coverages of the model
        :return: The coverages of the model
        """
        return self._coverages

    @property
    def loadings(self) -> pd.DataFrame:
        """
        This function returns the loadings of the model
        :return: The loadings of the model
        """
        return self._loadings

    @property
    def network(self) ->  MultiDiGraph:
        """
        This function returns the network of the model
        :return: The network of the model
        """
        if not hasattr(self, '_network'):

            self._network = MultiDiGraph()

            self._network.add_nodes_from(
                [
                    (
                        index,
                        {
                            'index': i,
                            'type': row['NodeType'],
                            **row.to_dict()
                        }
                    )
                    for i, (index, row) in  enumerate(self.nodes.iterrows())
                ]
            )

            self._network.add_edges_from([
                (row['From Node'], row['To Node'], {'name': index, 'type': row['Type']})
                for i, (index, row) in enumerate(self.links.iterrows())]
            )

        return self._network

    def add_node(self, node_id: str, node_type: str, node_attributes: Dict[str, Any]) -> None:
        """
        This function adds a node to the model
        :param node_id: The node id
        :param node_type: The node type
        :param node_attributes: The node attributes
        :return:
        """
        pass

    @property
    def links(self) -> gpd.GeoDataFrame:
        """
        This function returns the links of the model
        :return: The links of the model
        """
        return self._links

    def add_link(self, link_id: str, link_type: str, link_attributes: Dict[str, Any]) -> None:
        pass

    def split_conduit(self, conduit_id: str, insert_node_id: str, split_ratios: List[float]) -> List[str]:
        pass

    def change_node_type(self, node_id: str, node_type: str, node_attributes: Dict[str, Any]) -> None:
        pass

    def find_all_paths(self, start_node: str, end_node: str) -> List[List[str]]:
        pass

    def to_shp(self, shp_path: str) -> None:
        """
        This function writes the model to a shapefile
        :param shp_path: The path to the shapefile
        :return:
        """
        pass

    def to_geojson(self, geojson_path: str) -> None:
        """
        This function writes the model to a geojson
        :param geojson_path: The path to the geojson
        :return:
        """
        pass

    def save(self, model_path: str) -> None:
        """
        This function saves the model to the model path
        :param model_path: The path to the model
        :return:
        """
        with open(model_path, 'w') as swmm_file:
            swmm_file.write(f'title {self.title}\n')

    def write_title(self, swmm_file) -> None:
        """
        This function writes the title to the SWMM file
        :param swmm_file: The SWMM file
        :return:
        """
        pass

    def __read_model(self, model_path: str) -> SpatialSWMM:
        """
        This function reads the model from the model path
        :param crs: Coordinate reference system for the model
        :param model_path: The path to the model
        :return:
        """

        current_section = ""

        with open(model_path, 'r') as swmm_file:

            comments = []
            coordinates = {}
            vertices = {}
            polygons = {}

            for line in swmm_file:

                line = line.strip()

                if line == '':
                    continue

                if line.startswith("[") and line.endswith(']'):
                    current_section = line
                else:

                    tokens = re.split(pattern=r'\s+|\t+', string=line)

                    if current_section.upper() in '[TITLE]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            model.title_comments.append(line)

                    elif current_section.upper() in '[OPTIONS]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            self._options.loc[tokens[0]] = tokens[1]

                            if len(comments) > 0:
                                self._options.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = comments
                            else:
                                self._options.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            comments.clear()

                    elif current_section.upper() in '[RAINGAGES]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._raingages.loc[tokens[0], SpatialSWMM.RAINGAGES_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._raingages.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._raingages.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            self._raingages.loc[tokens[0], "geometry"] = Point(float(0.0), float(0.0))

                            comments.clear()

                    elif current_section.upper() in '[JUNCTIONS]':

                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._nodes.loc[tokens[0], SpatialSWMM.JUNCTION_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._nodes.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._nodes.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            self._nodes.loc[tokens[0], "NodeType"] = "JUNCTIONS"
                            comments.clear()

                    elif current_section.upper() in '[OUTFALLS]':

                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._nodes.loc[tokens[0], SpatialSWMM.OUTFALL_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._nodes.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._nodes.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            self._nodes.loc[tokens[0], "NodeType"] = "OUTFALLS"
                            comments.clear()

                    elif current_section.upper() in '[STORAGE]':

                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._nodes.loc[tokens[0], SpatialSWMM.STORAGE_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._nodes.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._nodes.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            self._nodes.loc[tokens[0], "NodeType"] = "STORAGE"
                            comments.clear()

                    elif current_section.upper() in '[SUBCATCHMENTS]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._sub_catchments.loc[tokens[0], SpatialSWMM.SUB_CATCHMENT_COLUMNS[i - 1]] = tokens[
                                    i]

                            if len(comments) > 0:
                                self._sub_catchments.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._sub_catchments.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            self._sub_catchments.loc[tokens[0], "SubCatchmentType"] = "SUBCATCHMENTS"
                            comments.clear()

                    elif current_section.upper() in '[SUBAREAS]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._sub_areas.loc[tokens[0], SpatialSWMM.SUB_AREA_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._sub_areas.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._sub_areas.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            comments.clear()

                    elif current_section.upper() in '[CONDUITS]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._links.loc[tokens[0], SpatialSWMM.CONDUIT_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._links.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._links.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            self._links.loc[tokens[0], "LinkType"] = "CONDUITS"
                            comments.clear()

                    elif current_section.upper() in '[ORIFICES]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._links.loc[tokens[0], SpatialSWMM.ORIFICE_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._links.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._links.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            self._links.loc[tokens[0], "LinkType"] = "ORIFICES"
                            comments.clear()

                    elif current_section.upper() in '[WEIRS]':

                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._links.loc[tokens[0], SpatialSWMM.WEIR_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._links.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._links.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            self._links.loc[tokens[0], "LinkType"] = "WEIRS"
                            comments.clear()

                    elif current_section.upper() in '[OUTLETS]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._links.loc[tokens[0], SpatialSWMM.OUTLET_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._links.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._links.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            self._links.loc[tokens[0], "LinkType"] = "OUTLETS"
                            comments.clear()

                    elif current_section.upper() in '[XSECTIONS]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._xsections.loc[tokens[0], SpatialSWMM.XSECTION_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._xsections.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._xsections.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            comments.clear()

                    elif current_section.upper() in '[POLLUTANTS]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._pollutants.loc[tokens[0], SpatialSWMM.POLLUTANTS_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._pollutants.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._pollutants.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            comments.clear()

                    elif current_section.upper() in '[LANDUSES]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            for i in range(1, len(tokens)):
                                self._landuses.loc[tokens[0], SpatialSWMM.LANDUSES_COLUMNS[i - 1]] = tokens[i]

                            if len(comments) > 0:
                                self._landuses.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = ','.join(comments)
                            else:
                                self._landuses.loc[tokens[0], SpatialSWMM.COMMENTS_COLUMNS[0]] = None

                            comments.clear()

                    elif current_section.upper() in '[COORDINATES]':

                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            coordinates[tokens[0]] = (float(tokens[1]), float(tokens[2]))
                            comments.clear()

                    elif current_section.upper() in '[VERTICES]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            identifier = tokens[0]
                            if identifier in vertices:
                                vertices[identifier].append((float(tokens[1]), float(tokens[2])))
                            else:
                                vertices[identifier] = [(float(tokens[1]), float(tokens[2]))]

                    elif current_section.upper() in '[POLYGONS]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            identifier = tokens[0]
                            if identifier in polygons:
                                polygons[identifier].append((float(tokens[1]), float(tokens[2])))
                            else:
                                polygons[identifier] = [(float(tokens[1]), float(tokens[2]))]
                    elif current_section.upper() in '[SYMBOLS]':
                        if line.startswith(';;'):
                            pass
                        elif line.startswith(';'):
                            comments.append(line)
                        else:
                            coordinates[tokens[0]] = (float(tokens[1]), float(tokens[2]))

        for raingage_id, _ in self._raingages.iterrows():
            if raingage_id in coordinates:
                self._raingages.loc[raingage_id, 'geometry'] = Point(coordinates[raingage_id])

        for node_id, _ in self._nodes.iterrows():
            if node_id in coordinates:
                self._nodes.loc[node_id, 'geometry'] = Point(coordinates[node_id])

        for link_id, _ in self._links.iterrows():
            if link_id in vertices:
                vertex_list = vertices[link_id]
                start_node_column = self._links.columns[1]
                end_node_column = self._links.columns[2]

                start_node = self._links.loc[link_id, start_node_column]
                end_node = self._links.loc[link_id, end_node_column]

                vertex_list = [coordinates[start_node]] + vertex_list + [coordinates[end_node]]
                self._links.loc[link_id, 'geometry'] = LineString(vertex_list)
            else:
                start_node_column = self._links.columns[1]
                end_node_column = self._links.columns[2]
                start_node = self._links.loc[link_id, start_node_column]
                end_node = self._links.loc[link_id, end_node_column]
                self._links.loc[link_id, 'geometry'] = LineString([coordinates[start_node], coordinates[end_node]])

        for sub_catchment_id, _ in self._sub_catchments.iterrows():
            if sub_catchment_id in polygons:
                polygon_list = polygons[sub_catchment_id]
                self._sub_catchments.loc[sub_catchment_id, 'geometry'] = Polygon(polygon_list)

