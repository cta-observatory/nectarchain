# nectarcam

Repository for the high level analysis of the NectarCAM.  
The analysis is heavily based on [ctapipe](https://github.com/cta-observatory/ctapipe).


## Installation

At this moment, we depend on Franca's nectarcamreader fork of ctapipe, which includes `ctapipe.io.nectarcameventsource`.

```
git clone https://github.com/cta-observatory/ctapipe
git remote add nectarcamreader https://github.com/FrancaCassol/ctapipe.git
git fetch nectarcamreader
git checkout -b nectar nectarcamreader/nectarCAM_reader
```

Note that `protozfitsreader` should be installed (see https://github.com/cta-sst-1m/protozfitsreader) because `ctapipe.io.nectarcameventsource` depends on it to read Zfits files.


## Contributing

All contribution are welcomed.

Guidelines are the same as [ctapipe's ones](https://cta-observatory.github.io/ctapipe/development/index.html)
See [here](https://cta-observatory.github.io/ctapipe/development/pullrequests.html) how to make a pull request to contribute.


## Report issue / Ask a question

Use GitHub Issues.
