%% exampele to build a keras neural network in matlab via python bridge

%------ works for Linux/Mac OS only ------%

% bypass matlab built-in hdf5 implementation
% see: https://www.mathworks.com/matlabcentral/answers/265247-importing-custom-python-module-fails#comment_338642
RTLD_NOW = 2;
RTLD_DEEPBIND = 8;
flag = bitor(RTLD_NOW, RTLD_DEEPBIND); % RTLD_NOW | RTLD_DEEPBIND
py.sys.setdlopenflags(int32(flag));

% import modules
tf = py.importlib.import_module('tensorflow');
np = py.importlib.import_module('numpy');

% import used layers, Dense and Sequential
Dense = tf.keras.layers.Dense;
Sequential = tf.keras.Sequential;

% construct a simple nn model
% NOTE: use `py.list({ })` to mimic python list
model = Sequential(py.list({...
    Dense(int32(64), pyargs("activation", tf.nn.relu)), ...
    Dense(int32(64), pyargs("activation", tf.nn.relu)), ...
    Dense(int32(1), pyargs("activation", tf.nn.sigmoid)) ...
    }));

% compile model
model.compile(pyargs("optimizer", "adam", ...
    "loss", "binary_crossentropy", ...
    "metrics", py.list({'accuracy'})))

% prepare dataset
x_train = np.random.rand(int32(10000), int32(2));
x_train_matlab = double(x_train);
y_train_matlab = mean(x_train_matlab, 2) < 0.5;
% y_train = py.list({});
% for k = 1 : double(py.len(x_train))
%     if y_train_matlab(k)
%         tmp = np.asarray(py.list({ int32(1)}));
%     else
%         tmp = np.asarray(py.list({ int32(0)}));
%     end
%     y_train.append(tmp);
% end
% y_train = np.asarray(y_train).reshape(py.list({py.len(x_train), int32(1)}));
y_train = array2numpy(double(y_train_matlab));

% training
model.fit(x_train, y_train, pyargs("epochs", int32(5)))