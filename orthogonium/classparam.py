from typing import Callable


class ClassParam:
    def __init__(self, fct: Callable = None, *args, **kwargs):
        """
        ClassParam is a wrapper which eases the customization of models in the model
        factory. It allows to customize layer types and layer default params,
        with a lot of flexibility.

        How it works:
            It works as a partial function: the user provides a function and sets the
            default args/kwargs. Those args will be used when calling the
            `ClassParam` but args added at call will override the default ones.

            .. highlight:: python
            .. code-block:: python

                # some basic use
                def function(a, b, c):
                    return a, b, c

                cp = ClassParam(function, 1, 2, c=3)
                cp() == (1, 2, 3)
                cp('a') == ('a', 2, 3)
                cp('a', c='c') == ('a', 2, 'c')

                # a more realistic use
                linear = ClassParam(layers.Dense, use_bias=True, activation='relu')
                linear(64) # linear with 64 neurons, bias and relu
                linear(64, activation=None) # similar but with no activation

        Args:
            fct: Callable the function to be called with default arguments
            *args: default args, the n args provided at call will overwrite the n
                first args provided at init.
            **kwargs: default kwargs, kwargs provided during call will overwrite the
                args provided at init.

        Warning:
            An arg provided in the args fashion during init cannot be overwritten by
            a kwarg at call

            .. highlight:: python
            .. code-block:: python

                def function(a, b, c):
                        return a, b, c

                cp = ClassParam(function, 1, 2, c=3)
                cp(b='b') # fails
                cp2 = ClassParam(function, 1, b=2, c=3)
                cp2(b='b') # works
        """
        self.fct = fct
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        """
        Call self.fct with parameter given in self.kwargs and self.args
        The parameter given when calling this function are merged with the ones
        given at instantiation. Parameter given at call override automatically the
        one given at instantiation.
        """
        if self.args is not None:
            args = args + self.args[len(args) :]
        if self.kwargs is not None:
            kwargs = dict(list(self.kwargs.items()) + list(kwargs.items()))
        if self.fct is not None:
            return self.fct(*args, **kwargs)
        else:
            return lambda x: x

    def __str__(self):
        if self.args is not None:
            args_list = list(map(str, self.args))
        else:
            args_list = []
        if self.kwargs is not None:
            kwargs_list = [f"{k}={str(v)}" for k, v in self.kwargs.items()]
        else:
            kwargs_list = []
        return f"{self.fct.__name__}({','.join(args_list+kwargs_list)})"

    def get_config(self, flatten=False):
        """
        Return a dict containing the config of this classparam. This differs from
        tf.get_config as the dict cannot be reused as is to reinstantiate the
        classparam.

        Args:
            flatten: when false, recurse in the *args and **kwargs to build a nested
                structure, else nested structure are converted to string to get a
                single level dict.

        Returns: a dict describing the classparam

        """

        if self.args is not None:
            args_dict = dict(
                (f"args_{i}", _pretty_print(elt)) for i, elt in enumerate(self.args)
            )
        else:
            args_dict = None
        if self.kwargs is not None:
            kwargs_dict = dict((k, _pretty_print(v)) for k, v in self.kwargs.items())
        else:
            kwargs_dict = None
        if flatten:
            ret = {"fct": self.fct.__name__}
            ret.update(args_dict)
            ret.update(kwargs_dict)
            return flatten_config(ret)
        return dict(fct=self.fct.__name__, args=args_dict, kwargs=kwargs_dict)


def _pretty_print(val):
    if isinstance(val, ClassParam):
        ret = val.get_config(flatten=True)
    elif hasattr(val, "get_config"):
        try:
            ret = val.get_config()
        except TypeError as te:
            ret = val().get_config()
    elif isinstance(val, Callable):
        if hasattr(val, "__name__"):
            ret = val.__name__
        else:
            ret = val.__class__.__name__
    elif isinstance(val, list):
        ret = [_pretty_print(x) for x in val]
    elif isinstance(val, dict):
        ret = dict((k, _pretty_print(v)) for k, v in val.items())
    else:
        ret = str(val)
    return ret


def flatten_config(nested_config, prepend_key=[]):
    """
    Take a list, dict or ClassParam and return the corresponding flattened
    configuration dictionnary. Every item in the dict are strings or int/float,
    making is usage as-is in a WandB configuration.


    .. highlight:: python
    .. code-block:: python

        flattened_config = flatten_config(wrn.non_lip_layers_params)
        flattened_config == {
            'conv/fct': 'Conv2D',
            'conv/kernel_size': '(3, 3)',
            'conv/padding': 'same',
            'conv/use_bias': 'True',
            'dropout/fct': 'Dropout',
            'dropout/args_0': '0.1'
        }


    Args:
        nested_config: object to flatten, can be a list, a dict or a ClassParam.
        prepend_key: parameter used for recursive call (adds a string at the
            beginning of each key of the output). No need to be set.

    Returns:
        a flattened dict with the config.

    """
    if isinstance(nested_config, (dict, list)):
        if isinstance(nested_config, list):
            nested_config = dict(
                zip(
                    [f"args_{i}" for i in range(len(nested_config))],
                    nested_config,
                )
            )
        out_dict = {}
        for k, v in nested_config.items():
            out_dict.update(flatten_config(_pretty_print(v), prepend_key + [k]))
        return out_dict
    elif isinstance(nested_config, ClassParam):
        return nested_config.get_config(flatten=True)
    else:
        return {"/".join(prepend_key): nested_config}
