# Model-Pruning
Model pruning is one of the most esssential steps 
Lets go understanding the essence of it

Most of the weights in a neural network are not required.
For example on ImageNet Dataset which is widely used 
1. In ResNet 50, 90% sparse matches the baseline accuracy
2. In MobileNet which is extensively used, 75% sparse matches the baseline accuracy

Comparison between unpruned model and pruned model
Unpruned: ResNet 50, 4 years earlier, used to take 100 milli seconds per image. ( Hardware can also be considered to an extent)
Pruned  : Today, using pruning, reducing the denseLayers 2x times, ResNet50 can now perform in 10 milli seconds per image.
Unpruned: VGG 16 is 500MB
Pruned  : Today by pruning, it is reduced to 11.3 MB which is 49X times compressed.

Reasons for industries not taking advantage of Pruning
1. Frankly pruning is a little iterative process and requries lots and lots of intuition
2. Resources to manage becomes expensive
3. Systems and the hardware are not still feasible enough for performing unstructured pruning for speedup
4. Lack of support and ease of use in ML frameworks.

Then are we not using Pruning or how are we using Pruning
Firstly realizing this requirement in the market had made tensorflow and pytorch pioneers in ML frameworks.
Tensorflow had built great API's using keras for automatically implement pruning just by some attributes and hyperparameters tuning.
This helped most of the developers to reduce their model size and weight and deploy faster execution in realtime

In the file, there is a modified code of modelpruning using tensorflow sourced from tensorflow.org
Will be uploading model pruning using pytorch framework and libraries shortly.
