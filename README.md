![banana](https://i2.wp.com/www.anthropocenemagazine.org/wp-content/uploads/2019/08/banana.jpg?zoom=2)

# PerfectBanana - WIP

> Android app for classifying level of ripeness of a banana, using photo as input.

> built using TensorFlow in Python 3.7 and Android Studio in Java.

**In order to run predict_ripeness.py, pop this:**
```shell
$ python predict_ripeness.py -i data/test_images/x.jpg -m output/vggnet.model -l output/vggnet_lb.pickle -w 64 -he 64
```
**In order to run build_conv_nn.py, pop this:**
```shell
$ python build_conv_nn.py -d data -m output/vggnet.model -l output/vggnet_lb.pickle -p output/vggnet_plot.png
```
