# Running the code

To train the model download the dataset from https://fki.tic.heia-fr.ch/databases/iam-handwriting-database then run</br>

```python model.py```



To run code on a pretrained model </br>
```python main.py <image file path> <image type> <model file path> <translation language> <verbose>```</br>

The following are examples to run on images in the experiment images folder </br>

```python main.py ../experiment_imgs/bill_multi.png 0 ../trained_models/model.h5 de 1```</br>
```python main.py ../experiment_imgs/soup8.png 0 ../trained_models/model.h5 de 0```</br>
```python main.py ../experiment_imgs/multi_line.png 0 ../trained_models/model.h5 de 0```</br>
```python main.py ../experiment_imgs/his2.png 2 ../trained_models/model.h5 de 0```</br>
```python main.py ../experiment_imgs/disturbance.png 0 ../trained_models/model.h5 de 0```</br>