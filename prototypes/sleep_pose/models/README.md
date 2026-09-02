# Eye-state model

`open-closed-eye-0001.onnx` is the Open Model Zoo open/closed-eye classifier.

- Source: <https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/public/open-closed-eye-0001>
- Original model: <https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/open-closed-eye-0001/open_closed_eye.onnx>
- License: Apache License 2.0; redistribution copy: [LICENSE.open-closed-eye-0001.txt](LICENSE.open-closed-eye-0001.txt)
- SHA-384: `2615bce53b55620c629db21b043057600ccc53466f053c0a8277c43577c2db21e48f330cf9b15213016d17cddb8cba27`

The model accepts a `1x3x32x32` BGR tensor normalized with mean 127 and
scale 255, and returns `[open, closed]` probabilities.
