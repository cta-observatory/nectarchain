# NectaRTA

Repository for the Bokeh webpage application used for the visualisation of the Real Time Analysis of the NectarCAM camera of the MST of the [CTAO](https://www.ctao.org/).
The application displays different quantities from R0 format of data to DL3. The frontent relies on the RTA pipeline (*insert link when available*).
The goal is to merge it with the [nectarchain](https://github.com/cta-observatory/nectarchain/tree/main) pipeline of the [CTAO Consortium](https://github.com/cta-observatory) repository.

## Pipeline usage

To run the available pipeline:

- Download the repository
- Run the Bokeh webpage using:
```
bokeh serve bokeh_app --show --dev
```

The test suite is provided in ``tests/test.py``. It can be run as such and will use example DL1 stored in ``example_data``.
If you want to run it from the ``bokeh`` command, use:
```
bokeh serve bokeh_app --show --dev --args test-interface
```

**None**: Be careful to have all the dependencies installed in your environment.

**Note**: This is an *alpha* version of the webpage that still needs to be modified. It might not be 100% stable.

## Report issue

For any issue, please contact directly the developer at julian.hamo@ijclab.in2p3.fr.
