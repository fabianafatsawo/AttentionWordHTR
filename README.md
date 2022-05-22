# AttentionHTR
Handwriting recognition model by word segmentation

Install dependencies

```
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

Download pre-trained models from [here](https://drive.google.com/drive/folders/1h6edewgRUTJPzI81Mn0eSsqItnk9RMeO?usp=sharing). 

Step 1:
Download the ```AttentionHTR-General-sensitive.pth``` model and place it into ```AttentionHTR/model/saved_models```.

Step 2:
Predict!

Predict single image and save result.

```
python3 main.py --image "images/phone4.png" --save 1
```

Predict directory with images.

```
python3 main.py --image "images/"
```

Tweak settings
```
python3 main.py --image "images/img.jpeg" --line_dilation 1 --word_dilation 1
```
