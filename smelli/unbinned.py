from collections import OrderedDict
from flavio.io import instanceio as iio
import numpy as np
import yaml
from .util import get_cachepath
from requests import get

class UnbinnedParameterLikelihood(iio.YAMLLoadable):
    """An `UnbinnedParameterLikelihood` provides an unbinned likelihood function in terms of
    parameters.
    Methods:
    - `log_likelihood`: The likelihood as a function of the parameters
    Instances can be imported and exported from/to YAML using the `load`
    and `dump` methods.
    """
    _input_schema_dict = {
        'eft': str,
        'basis': str,
        'scale': float,
        'names': list,
        'samples': np.array,
        'weights': np.array
    }

    _output_schema_dict = {
        'eft': str,
        'basis': str,
        'scale': float,
        'names': list,
        'samples': np.array,
        'weights': np.array
    }

    def __init__(self,
                 eft,
                 basis,
                 scale,
                 names,
                 samples,
                 weights=None):
        """Initialize the instance.
        Parameters:
        - `eft`: a string identifying the WET as defined by wcxf
        - `basis`: a string identifying the WET basis as defined by wcxf
        - `scale`: a scale (in units of GeV) at which the WET coefficient samples are valid
        - `names`: an ordered list of names identiying the Wilson coefficients in the WET basis
        - `samples`: a numpy array of vectors of Wilson coefficients, with each row representing a sample
        - `weights`: a numpy array of scalar weights on a logarithmic scale for each sample
        """
        self.eft     = eft
        self.basis   = basis
        self.scale   = scale
        self.names   = names
        self.samples = samples
        self.weights = weights if weights is not None else np.ones(len(samples)) / len(samples)

    def log_likelihood(self, smeft_wc):
        """Return the likelihood for all parameters."""

        wet_wc = smeft_wc.match_run(self.scale, self.eft, self.basis)
        central = [wet_wc[name] for name in self.names]
        result = 0.0
        for weight, sample in zip(self.weights, self.samples):
            result += weight * np.sum(np.absolute(sample - central)**2)

        return result

    @classmethod
    def load(cls, f, **kwargs):
        """Instantiate an object from a YAML string or stream."""
        d = yaml.load(f, Loader=yaml.Loader)
        return cls.load_dict(d, **kwargs)

    def get_yaml_dict(self):
        """Dump the object to a YAML dictionary."""
        d = self.__dict__.copy()
        schema = self.output_schema()
        d = schema(d)
        # remove NoneTypes and empty lists
        d = {k: v for k, v in d.items() if v is not None}
        return d

class DataSets(iio.YAMLLoadable):
    """`DataSets` provides an interface to the list of data sets used for unbinned likelihood functions.
    Methods:
    - `log_likelihood`: The likelihood as a function of the parameters
    Instances can be imported and exported from/to YAML using the `load`
    and `dump` methods.
    """
    _input_schema_dict = {
        'datasets': OrderedDict,
    }

    _output_schema_dict = {
        'datasets': OrderedDict,
    }

    def __init__(self, datasets):
        self.datasets = datasets

    @classmethod
    def load(cls, f, **kwargs):
        """Instantiate an object from a YAML string or stream."""
        d = yaml.load(f, Loader=yaml.Loader)
        return cls.load_dict(d, **kwargs)

    def download(self, dn):
        if dn not in self.datasets.keys():
            raise ValueError('unknown dataset {}'.format(dn))

        destdir = os.path.join(get_cachepath, dn)
        if not os.path.exists(destdir):
            os.makedirs(destdir)

        for url in self.datasets[dn].urls:
            fn = os.path.join(destdir, url[url.rfind('/') + 1:])
            content = get(url).content
            with open(fn, 'w+') as f:
                write(f, content)
