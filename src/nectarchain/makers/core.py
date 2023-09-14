import logging
logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)
log.handlers = logging.getLogger('__main__').handlers

from abc import ABC, abstractmethod

from ctapipe_io_nectarcam import NectarCAMEventSource

from ..data import DataManagement

__all__ = ["BaseMaker"]

class BaseMaker(ABC):
    """Mother class for all the makers, the role of makers is to do computation on the data. 
    """
    @abstractmethod
    def make(self, *args, **kwargs):
        """
        Abstract method that needs to be implemented by subclasses.
        This method is the main one, which computes and does the work. 
        """
        pass
    @staticmethod
    def load_run(run_number : int,max_events : int = None, run_file = None) : 
        """Static method to load from $NECTARCAMDATA directory data for specified run with max_events

        Args:self.__run_number = run_number
            run_number (int): run_id
            maxevents (int, optional): max of events to be loaded. Defaults to 0, to load everythings.
            run_file (optional) : if provided, will load this run file
        Returns:
            List[ctapipe_io_nectarcam.NectarCAMEventSource]: List of EventSource for each run files
        """
        if run_file is None : 
            generic_filename,_ = DataManagement.findrun(run_number)
            log.info(f"{str(generic_filename)} will be loaded")
            eventsource = NectarCAMEventSource(input_url=generic_filename,max_events=max_events)
        else :  
            log.info(f"{run_file} will be loaded")
            eventsource = NectarCAMEventSource(input_url=run_file,max_events=max_events)
        return eventsource