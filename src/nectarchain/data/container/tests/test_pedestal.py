import numpy as np
import tempfile
from ctapipe.io import HDF5TableWriter
from nectarchain.data.container import NectarCAMPedestalContainer

def generate_mock_pedestal_container():
    # fixed values
    nchannels = 2
    npixels = 10
    nevents_max = 100
    nevents_min = 10
    nsamples = np.uint8(60)
    # random values for other fields
    nevents = np.float64(np.random.randint(nevents_min, nevents_max, size=(npixels,)))
    pixels_id = np.array([2, 4, 3, 8, 6, 9, 7, 1, 5, 10], dtype=np.uint16)
    ucts_timestamp_min = np.uint64(np.random.randint(1e8))
    ucts_timestamp_max = np.uint64(np.random.randint(1e8 + 100))
    pedestal_mean_hg = np.float64(np.random.uniform(240, 260, size=(npixels, nsamples)))
    pedestal_mean_lg = np.float64(np.random.uniform(240, 260, size=(npixels, nsamples)))
    pedestal_std_hg = np.float64(np.random.normal(size=(npixels, nsamples)))
    pedestal_std_lg = np.float64(np.random.normal(size=(npixels, nsamples)))
    pixel_mask = np.int8(np.random.randint(0,2,size=(nchannels,npixels)))

    # create pedestal container
    pedestal_container = NectarCAMPedestalContainer(
        nsamples=nsamples,
        nevents=nevents,
        pixels_id=pixels_id,
        ucts_timestamp_min=ucts_timestamp_min,
        ucts_timestamp_max=ucts_timestamp_max,
        pedestal_mean_hg=pedestal_mean_hg,
        pedestal_mean_lg=pedestal_mean_lg,
        pedestal_std_hg=pedestal_std_hg,
        pedestal_std_lg=pedestal_std_lg,
        pixel_mask = pixel_mask
    )
    pedestal_container.validate()

    # create dictionary that duplicates content
    dict = {'nsamples': nsamples,
            'nevents': nevents,
            'pixels_id': pixels_id,
            'ucts_timestamp_min': ucts_timestamp_min,
            'ucts_timestamp_max': ucts_timestamp_max,
            'pedestal_mean_hg': pedestal_mean_hg,
            'pedestal_mean_lg': pedestal_mean_lg,
            'pedestal_std_hg': pedestal_std_hg,
            'pedestal_std_lg': pedestal_std_lg,
            'pixel_mask': pixel_mask
            }

    # return both container and input content
    return pedestal_container, dict

class TestNectarCAMPedestalContainer:

    def test_create_pedestal_container_mem(self):
        # create mock pedestal container
        pedestal_container, dict = generate_mock_pedestal_container()

        # check that all fields are filled correctly with input values
        assert pedestal_container.nsamples == dict['nsamples']
        assert pedestal_container.nevents.tolist() == dict['nevents'].tolist()
        assert pedestal_container.ucts_timestamp_min == dict['ucts_timestamp_min']
        assert pedestal_container.ucts_timestamp_max == dict['ucts_timestamp_max']
        assert np.allclose(pedestal_container.pedestal_mean_hg, dict['pedestal_mean_hg'])
        assert np.allclose(pedestal_container.pedestal_mean_lg, dict['pedestal_mean_lg'])
        assert np.allclose(pedestal_container.pedestal_std_hg, dict['pedestal_std_hg'])
        assert np.allclose(pedestal_container.pedestal_std_lg, dict['pedestal_std_lg'])
        assert np.allclose(pedestal_container.pixel_mask, dict['pixel_mask'])

    # FIXME
    # Guillaume is working on generalizing the fromhdf5 method to all containers
    # This test should work once it's done
    # def test_pedestal_container_io(self):
    #     input_pedestal_container, dict = generate_mock_pedestal_container()
    #     with tempfile.TemporaryDirectory() as tmpdirname:
    #         outpath = tmpdirname+"/pedestal_container_0.h5"
    #         writer = HDF5TableWriter(
    #             filename=outpath,
    #             mode="w",
    #             group_name="data",
    #         )
    #         writer.write(table_name="NectarCAMPedestalContainer",
    #                      containers=input_pedestal_container)
    #         writer.close()
    #
    #         pedestal_container = next(NectarCAMPedestalContainer.from_hdf5(outpath))
    #
    #         #check that container is an instance of the proper class
    #         assert isinstance(pedestal_container,NectarCAMPedestalContainer)
    #
    #         #check content
    #         assert pedestal_container.nsamples == dict['nsamples']
    #         assert pedestal_container.nevents.tolist() == dict['nevents'].tolist()
    #         assert pedestal_container.ucts_timestamp_min == dict['ucts_timestamp_min']
    #         assert pedestal_container.ucts_timestamp_max == dict['ucts_timestamp_max']
    #         assert np.allclose(pedestal_container.pedestal_mean_hg, dict['pedestal_mean_hg'])
    #         assert np.allclose(pedestal_container.pedestal_mean_lg, dict['pedestal_mean_lg'])
    #         assert np.allclose(pedestal_container.pedestal_std_hg, dict['pedestal_std_hg'])
    #         assert np.allclose(pedestal_container.pedestal_std_lg, dict['pedestal_std_lg'])

if __name__ == "__main__":
    pass
