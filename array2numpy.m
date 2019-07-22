function np_array = array2numpy(array)
%% convert matlab array to a numpy array

% import numpy package
np = py.importlib.import_module('numpy');

shape = size(array);
% conver to python tuple
shape_py = py.tuple(int32(shape));

% the order of numpy and matlab is different when dealing with reshape
% see: https://bic-berkeley.github.io/psych-214-fall-2016/index_reshape.html
% https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
np_array = np.asarray(py.list(array(:))).reshape(shape_py, ...
                                                 pyargs('order', 'F'));