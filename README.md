# jmyml
We learn sycl and machine learning algorithms from scratch together!

## build & run
You need `git lfs` if you want to work with the datasets provided.
This software is clearly dependent on a working sycl compiler.
Please install one, such as AdaptiveCpp or intel oneAPI DPC++.
Then copy `CMakeUserPresets.json.example` to `CMakeUserPresets.json` and set the compiler to the correct path.
Now you can build using:

```
cmake --preset sycl-BUILD_TYPE
cmake --build --preset sycl-BUILD_TYPE
```

where `BUILD_TYPE` is one of `debug` and `release`.

The executable binaries will be output to either `./bin/Debug` or `./bin/Release` depending on the BUILD_TYPE.
There are a few tests and the executable `IdxViewer`.
This can visualize images in the `idx` format used by the MNIST dataset.
You can call this executable for example by

```
bin/IdxViewer data/train-images.idx
```

and using your LEFT/RIGHT arrow keys you can view more images.