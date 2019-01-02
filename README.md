# Banana Recognizer 🍌

**Convolutional Neural Network:** `BananaNetwork/banana_net_17_12_18.h5`
* `INPUT > CONV > RELU > POOL > FC > RELU > FC`
* Build script: `CNN.py`
> `BananaNetwork/banana_net.h5 is currently overfitted`


**Convolutional Netural Network:** `BananaNetwork/le_banana_net.h5`

* `INPUT > CONV > RELU > POOL > CONV > RELU > POOL > FC > RELU > FC`
* Build script: `CNN_LeNet.py`

* Also looking at other alternatives for design of network

<!-- $ tree -v -L 2 --charset utf-8-->
##### File structure

```
├── BananaNetwork
│   ├── banana_net.h5
│   ├── banana_net_17_12_18.h5
│   └── le_banana_net.h5
├── CNN.py
├── CNN_LeNet.py
├── README.md
├── WebcamApp
│   └── camera_app.py
├── banana_predicter.py
├── datasets
│   ├── google_images
│   │   └── ...
│   ├── google_synset_fruit365
│   │   └── ...
│   └── synset_fruit365
│       └── ...
├── epoch_fig.png
├── google_images_scraper.js
├── gui
│   ├── gui.py
│   └── gui.ui
├── image_downloader.py
├── network_trainer.py
├── predict_tests
│   └── ...
└── urls
    ├── google_banana_urls.txt
    └── synset_banana_urls.txt
```

#### Datasets

* Fruit365
* Synset - ImageNet
* Google Images

#### Latest training

![EpochFig](epoch_fig.png)
