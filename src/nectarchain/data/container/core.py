import copy
import importlib
import logging
from pathlib import Path

import numpy as np
from ctapipe.containers import Container, EventType, Field, Map, partial
from ctapipe.core.container import FieldValidationError
from ctapipe.io import HDF5TableReader
from tables.exceptions import NoSuchNodeError

logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)
log.handlers = logging.getLogger("__main__").handlers


__all__ = [
    "ArrayDataContainer",
    "TriggerMapContainer",
    "get_array_keys",
    "merge_map_ArrayDataContainer",
]


def get_array_keys(container: Container):
    """Return a list of keys corresponding to fields which are array type in the given
    container.

    Parameters:
        container (Container): The container object to search for array fields.

    Returns:
        list: A list of keys corresponding to array fields in the container.

    Example:
        >>> container = Container()
        >>> container.field1 = np.array([1, 2, 3])
        >>> container.field2 = 5
        >>> container.field3 = np.array([4, 5, 6])
        >>> get_array_keys(container)
        ['field1', 'field3']
    """
    keys = []
    for key, field in container.fields.items():
        if field.type == np.ndarray:
            keys.append(key)
    return keys


class NectarCAMContainer(Container):
    """Base class for the NectarCAM containers.

    This container cannot be recursive, to be directly written with a HDF5TableWriter.
    """

    @staticmethod
    def _container_from_hdf5(path, container_class, index_component=0):
        """
        Static method to read a container from an HDF5 file.

        Parameters:
        path (str or Path): The path to the HDF5 file.
        container_class (Container): The class of the container to be filled with data.

        Yields:
        Container: The container from the data in the HDF5 file.

        Example:
        >>> container = NectarCAMContainer._container_from_hdf5('path_to_file.h5',
        MyContainerClass)
        """
        if isinstance(path, str):
            path = Path(path)

        container = container_class()
        with HDF5TableReader(path) as reader:
            tableReader = reader.read(
                table_name=f"/data/{container_class.__name__}_{index_component}",
                containers=container_class,
            )
            container = next(tableReader)

        yield container

    @classmethod
    def from_hdf5(cls, path, index_component=0):
        """
        Reads a container from an HDF5 file.

        Parameters:
        path (str or Path): The path to the HDF5 file.

        This method will call the _container_from_hdf5 method with the container
          argument associated to its own class (ArrayDataContainer)

        Yields:
        Container: The container generator linked to the HDF5 file.

        Example:
        >>> container = NectarCAMContainer.from_hdf5('path_to_file.h5')
        """

        return cls._container_from_hdf5(
            path, container_class=cls, index_component=index_component
        )


class ArrayDataContainer(NectarCAMContainer):
    """A container that holds information about waveforms from a specific run.

    Attributes:
        run_number (int): The run number associated with the waveforms.
        nevents (int): The number of events.
        npixels (int): The number of pixels.
        camera (str): The name of the camera.
        pixels_id (np.ndarray): An array of pixel IDs.
        broken_pixels_hg (np.ndarray): An array of high gain broken pixels.
        broken_pixels_lg (np.ndarray): An array of low gain broken pixels.
        ucts_timestamp (np.ndarray): An array of events' UCTS timestamps.
        ucts_busy_counter (np.ndarray): An array of UCTS busy counters.
        ucts_event_counter (np.ndarray): An array of UCTS event counters.
        event_type (np.ndarray): An array of trigger event types.
        event_id (np.ndarray): An array of event IDs.
        trig_pattern_all (np.ndarray): An array of trigger patterns.
        trig_pattern (np.ndarray): An array of reduced trigger patterns.
        multiplicity (np.ndarray): An array of events' multiplicities.
    """

    run_number = Field(
        type=np.uint16,
        description="run number associated to the waveforms",
    )
    nevents = Field(
        type=np.uint64,
        description="number of events",
    )
    npixels = Field(
        type=np.uint16,
        description="number of effective pixels",
    )
    pixels_id = Field(type=np.ndarray, dtype=np.uint16, ndim=1, description="pixel ids")
    broken_pixels_hg = Field(
        type=np.ndarray, dtype=bool, ndim=2, description="high gain broken pixels"
    )
    broken_pixels_lg = Field(
        type=np.ndarray, dtype=bool, ndim=2, description="low gain broken pixels"
    )
    camera = Field(
        type=str,
        description="camera name",
    )
    ucts_timestamp = Field(
        type=np.ndarray, dtype=np.uint64, ndim=1, description="events ucts timestamp"
    )
    ucts_busy_counter = Field(
        type=np.ndarray, dtype=np.uint32, ndim=1, description="ucts busy counter"
    )
    ucts_event_counter = Field(
        type=np.ndarray, dtype=np.uint32, ndim=1, description="ucts event counter"
    )
    event_type = Field(
        type=np.ndarray, dtype=np.uint8, ndim=1, description="trigger event type"
    )
    event_id = Field(type=np.ndarray, dtype=np.uint32, ndim=1, description="event ids")
    trig_pattern_all = Field(
        type=np.ndarray, dtype=bool, ndim=3, description="trigger pattern"
    )
    trig_pattern = Field(
        type=np.ndarray, dtype=bool, ndim=2, description="reduced trigger pattern"
    )
    multiplicity = Field(
        type=np.ndarray, dtype=np.uint16, ndim=1, description="events multiplicity"
    )


class TriggerMapContainer(Container):
    """Class representing a TriggerMapContainer.

    This class inherits from the `Container` class and is used to store trigger mappings
    of containers.

    Attributes:
        containers (Field): A field representing the trigger mapping of containers.

    Methods:
        is_empty(): Checks if the TriggerMapContainer is empty.
        validate(): Validates the TriggerMapContainer by checking if all the containers
        mapped are filled by correct type.

    Example:
        >>> container = TriggerMapContainer()
        >>> container.is_empty()
        True
        >>> container.validate()
        None
    """

    containers = Field(
        default_factory=partial(Map, Container),
        description="trigger mapping of Container",
    )

    @classmethod
    def from_hdf5(cls, path, slice_index=None, index_component=0):
        """
        Reads a container from an HDF5 file.

        Parameters:
        path (str or Path): The path to the HDF5 file.
        slice_index (int, optional): The index of the slice of data within the hdf5 file
        to read. Default is None.This method will call the _container_from_hdf5 method
        with the container argument associated to its own class (ArrayDataContainer)

        Yields:
        Container: The container generator linked to the HDF5 file.

        Example:
        >>> container = ArrayDataContainer.from_hdf5('path_to_file.h5')
        """

        return cls._container_from_hdf5(
            path,
            slice_index=slice_index,
            container_class=cls,
            index_component=index_component,
        )

    @staticmethod
    def _container_from_hdf5(
        path, container_class, slice_index=None, index_component=0
    ):
        """
        Reads a container from an HDF5 file.

        Parameters:
        path (str or Path): The path to the HDF5 file.
        container_class (Container): The class of the container to be read.
        slice_index (int, optional): The index of the slice of data within the hdf5 file
        to read. Default is None.

        This method first checks if the path is a string and converts it to a Path
        object
        if it is.
        It then imports the module of the container class and creates an instance of the
        container class.

        If the HDF5 file contains more than one slice and no slice index is provided,
        it reads all slices and yields a generator of containers.
        If a slice index is provided, it reads only the specified slice and returns the
        container instance.

        Yields:
        Container: The container associated to the HDF5 file.

        Raises:
        NoSuchNodeError: If the specified node does not exist in the HDF5 file.
        Exception: If any other error occurs.

        Example:
        >>> container = ArrayDataContainer._container_from_hdf5('path_to_file.h5',
        MyContainerClass)
        """
        if isinstance(path, str):
            path = Path(path)
        module = importlib.import_module(f"{container_class.__module__}")  # noqa :F841
        container = eval(f"module.{container_class.__name__}")()

        with HDF5TableReader(path) as reader:
            if len(reader._h5file.root.__members__) > 1 and slice_index is None:
                log.info(
                    f"reading {container_class.__name__} containing"
                    f"{len(reader._h5file.root.__members__)}"
                    f"slices, will return a generator"
                )
                for data in np.sort(reader._h5file.root.__members__):
                    # container.containers[data] =
                    # eval(f"module.{container_class.__name__}s")()

                    _container = eval(
                        f"module."
                        f"{container.fields['containers'].default_factory.args[0].__name__}"  # noqa
                    )
                    waveforms_data = eval(f"reader._h5file.root.{data}.__members__")
                    _mask = [_container.__name__ in _word for _word in waveforms_data]
                    _waveforms_data = np.array(waveforms_data)[_mask]
                    if len(_waveforms_data) == 1:
                        if issubclass(_container, TriggerMapContainer) or issubclass(
                            _container, ArrayDataContainer
                        ):
                            for key, trigger in EventType.__members__.items():
                                try:
                                    tableReader = reader.read(
                                        table_name=f"/{data}/{_waveforms_data[0]}"
                                        f"/{trigger.name}",
                                        containers=_container,
                                    )
                                    container.containers[trigger] = next(tableReader)
                                except NoSuchNodeError as err:
                                    log.warning(err)
                                except Exception as err:
                                    log.error(err, exc_info=True)
                                    raise err
                        else:
                            tableReader = reader.read(
                                table_name=f"/{data}/{_waveforms_data[0]}",
                                containers=_container,
                            )
                            container.containers[data] = next(tableReader)
                    else:
                        log.info(
                            f"there is {len(_waveforms_data)} entry"
                            f"corresponding to a {container_class}"
                            f"table save, unable to load"
                        )

                yield container
            else:
                if slice_index is None:
                    log.info(
                        f"reading {container_class.__name__} containing"
                        f"a single slice,"
                        f"will return the {container_class.__name__} instance"
                    )
                    data = "data"
                else:
                    log.info(
                        f"reading slice {slice_index} of {container_class.__name__},"
                        f"will return the {container_class.__name__} instance"
                    )
                    data = f"data_{slice_index}"
                _container = eval(
                    f"module.{container.fields['containers'].default_factory.args[0].__name__}"  # noqa
                )
                if issubclass(_container, TriggerMapContainer) or issubclass(
                    _container, ArrayDataContainer
                ):
                    for key, trigger in EventType.__members__.items():
                        try:
                            _container = eval(
                                f"module.{container.fields['containers'].default_factory.args[0].__name__}"
                            )
                            waveforms_data = eval(
                                f"reader._h5file.root.{data}.__members__"
                            )
                            _mask = [
                                _container.__name__ in _word for _word in waveforms_data
                            ]
                            _waveforms_data = np.array(waveforms_data)[_mask]
                            if len(_waveforms_data) == 1:
                                tableReader = reader.read(
                                    table_name=f"/{data}/{_waveforms_data[0]}/{trigger.name}",
                                    containers=_container,
                                )
                                # container.containers[data].containers[trigger] = next(tableReader)
                                container.containers[trigger] = next(tableReader)

                            else:
                                log.info(
                                    f"there is {len(_waveforms_data)} entry corresponding to a {container_class} table save, unable to load"
                                )
                        except NoSuchNodeError as err:
                            log.warning(err)
                        except Exception as err:
                            log.error(err, exc_info=True)
                            raise err
                else:
                    tableReader = reader.read(
                        table_name=f"/{data}/{_container.__name__}_"
                        f"{index_component}",
                        containers=_container,
                    )
                    data = f"data_{slice_index}"
                for key, trigger in EventType.__members__.items():
                    try:
                        _container = eval(
                            f"module.{container.fields['containers'].default_factory.args[0].__name__}"
                        )
                        tableReader = reader.read(
                            table_name=f"/{data}/{_container.__name__}_{index_component}/{trigger.name}",
                            containers=_container,
                        )
                        container.containers[trigger] = next(tableReader)
                    except NoSuchNodeError as err:
                        log.warning(err)
                    except Exception as err:
                        log.error(err, exc_info=True)
                        raise err
                yield container

    def is_empty(self):
        """This method check if the container is empty.

        Returns:
            bool: True if the container is empty, False otherwise.
        """
        return len(self.containers.keys()) == 0

    def validate(self):
        """Apply the validate method recursively to all the containers that are mapped
        within the TriggerMapContainer.

        Raises:
            FieldValidationError: if one container is not valid.
        """
        super().validate()
        for i, container in enumerate(self.containers):
            if i == 0:
                container_type = type(container)
            else:
                if not (isinstance(container, container_type)):
                    raise FieldValidationError(
                        "all the containers mapped must have the same type to be merged"
                    )


def merge_map_ArrayDataContainer(triggerMapContainer: TriggerMapContainer):
    """Merge and map ArrayDataContainer.

    This function takes a TriggerMapContainer as input and merges the array fields of
    the containers mapped within the TriggerMapContainer. The merged array fields are
    concatenated along the 0th axis. The function also updates the 'nevents' field of
    the output container by summing the 'nevents' field of all the mapped containers.

    Parameters:
        triggerMapContainer (TriggerMapContainer): The TriggerMapContainer object
        containing the containers to be merged and mapped.

    Returns:
        ArrayDataContainer: The merged and mapped ArrayDataContainer object.

    Example:
        >>> triggerMapContainer = TriggerMapContainer()
        >>> container1 = ArrayDataContainer()
        >>> container1.field1 = np.array([1, 2, 3])
        >>> container1.field2 = np.array([4, 5, 6])
        >>> container1.nevents
        3
        >>> container2 = ArrayDataContainer()
        >>> container2.field1 = np.array([7, 8, 9])
        >>> container2.field2 = np.array([10, 11, 12])
        >>> container2.nevents
        3
        >>> triggerMapContainer.containers['container1'] = container1
        >>> triggerMapContainer.containers['container2'] = container2
        >>> merged_container = merge_map_ArrayDataContainer(triggerMapContainer)
        >>> merged_container.field1
        array([1, 2, 3, 7, 8, 9])
        >>> merged_container.field2
        array([ 4,  5,  6, 10, 11, 12])
        >>> merged_container.nevents
        6
    """
    triggerMapContainer.validate()
    log.warning(
        "TAKE CARE TO MERGE CONTAINERS ONLY IF PIXELS ID, RUN_NUMBER (OR ANY FIELD THAT\
            ARE NOT ARRAY) ARE THE SAME"
    )
    keys = list(triggerMapContainer.containers.keys())
    output_container = copy.deepcopy(triggerMapContainer.containers[keys[0]])
    for key in keys[1:]:
        for field in get_array_keys(triggerMapContainer.containers[key]):
            output_container[field] = np.concatenate(
                (output_container[field], triggerMapContainer.containers[key][field]),
                axis=0,
            )
        if "nevents" in output_container.fields:
            output_container.nevents += triggerMapContainer.containers[key].nevents
    return output_container


# class TriggerMapArrayDataContainer(TriggerMapContainer):
#    containers = Field(default_factory=partial(Map, ArrayDataContainer),
#                       description = "trigger mapping of arrayDataContainer"
#                       )
