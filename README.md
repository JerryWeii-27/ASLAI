Dataset link: https://data.mendeley.com/datasets/48dg9vhmyk/2

**Thigns to note:**
- Rotated all 'Z' images so that the finger points up. The original orientation in the dataset is wrong.
- Moreover, 'Z' is suppose to be a dynamic sign. The signer should move their finger in a 'Z' shape. However, image classification model cannot recognize that. Therefore, it is currently indistinguishable from 'G'.
- If all 'G' images are taken from another angle, with the hole formed by thumb and middle finger facing the camera, the model would be able to tell 'G' form 'Z'. 
- 'J' in this dataset is wrong, but if you do the wrong sign, it would be recognized.
- 'J' is also a dynamic sign, maybe that is why it is wrong.