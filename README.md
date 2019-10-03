![banana](https://i2.wp.com/www.anthropocenemagazine.org/wp-content/uploads/2019/08/banana.jpg?zoom=2)

# PerfectBanana - WIP

> Android app for classifying level of ripeness of a banana, using photo as input.

> built using TensorFlow in Python 3.7 and Android Studio in Java.

**In order to run banana_nn.py, pop this:**
```shell
$ python banana_nn.py -d bananas -m output/nn.model -l output/nn_lb.pickle -p output/nn_plot.png
```
**In order to run banana_predictor.py, pop this:**
```shell
$ python banana_predictor.py -i test_images/ripe.jpg -m output/nn.model -l output/nn_lb.pickle -w 32 -he 32 -f 1
```
**In order to run banana_cnn.py, pop this:**
```shell
$ python banana_cnn.py -d bananas -m output/vggnet.model -l output/vggnet_lb.pickle -p output/vggnet_plot.png
```
