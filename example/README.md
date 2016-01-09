## loadccv example

(1) Download a ccv model, e.g. VGG-D[1] designed for ImageNet 2012[2]:

```bash
# ~214MB (half-float precision version)
curl -LO http://static.libccv.org/image-net-2012-vgg-d.sqlite3
```

(2) Convert to Torch7 format:

```bash
# We use CUDA here. If you do not have a GPU use `--backend nn` instead
loadccv --softmax --backend cunn --verbose image-net-2012-vgg-d.sqlite3
```

(3) Classify a test image[3]:

```
luajit main.lua indri-960.jpg
```

You should obtain something like that:

    RESULTS (top-5):
    ----------------
    score = 0.999250: indri, indris, Indri indri, Indri brevicaudatus
    score = 0.000640: Madagascar cat, ring-tailed lemur, Lemur catta
    score = 0.000049: koala, koala bear, kangaroo bear, native bear, Phascolarctos cinereus
    score = 0.000016: gibbon, Hylobates lar
    score = 0.000012: howler monkey, howler

--

[1]: Very Deep Convolutional Networks for Large-Scale Image Recognition, Karen Simonyan, Andrew Zisserman, ICLR 2015.

[2]: `imgnet.words` taken from [ccv](https://raw.githubusercontent.com/liuliu/ccv/stable/samples/image-net-2012.words).

[3]: [original image](https://en.wikipedia.org/wiki/Indri) by Karen Coppock (license: [CC BY 3.0](http://creativecommons.org/licenses/by/3.0)). Modified version created as follow:

```bash
convert -gravity Center \
-crop 960x960+0+0 \
<(curl -s https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Indri_Andasibe.JPG/1280px-Indri_Andasibe.JPG) \
indri-960.jpg
```
