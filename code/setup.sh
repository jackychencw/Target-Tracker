pip install --user Cython
pip install --user contextlib2
pip install --user pillow
pip install --user lxml
pip install --user jupyter
pip install --user matplotlib

# If you are using mac
brew install protobuf

git clone --depth 1 https://github.com/tensorflow/models
protoc models/research/object_detection/protos/*.proto --python_out=.
sudo pip3 install models/research/.

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim