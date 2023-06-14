1. python basis
    - python basics
        - colab stuff
        - jupyter notebook stuff
        - python basics
            - lambda, map, filter
    - python class
        - iter class
        - docstring
    - file, pandas, plotting
        - file
        - numpy
        - plot
        - seaborn
        - pandas

2. more python
    - colab drive file upload
    - pandas
    - vectors
        - quiver
    - debugging

3. mp, perceptron neurons
    - dataset
        - naming convention
        - sklearn dataset
        - class balance
        - train test split
        - main steps
        - plotting
    - mp neuron
        - dataset, binarise
        - inference
        - accuracy
        - test data validation
        - MP neruon class
    - perceptron neruon
        - perceptron
        - multiple epochs
        - checkpointing
        - learning rate
        - visualizing weights
        - matplotlib animation
        - HTML diplay

4. sigmoid neurons
    - plotting sigmoid
        - 1 input
        - 2 input
        - 3d plot, meshgrid
        - contour plot
        - custom color map
    - loss (square error)
        - loss, loss surface
        - lexicographic index to 2d index
    - sigmoid neuron class
        - coding tips
        - model
        - visualizing model-function
    - data standardisation, mapping
        - classification task using real ouput
        - standardisation of data
        - linear maping of data 
        - numpy reshaping
    - loss plot, tqdm
        - data preprocessing
        - training sigmoid neuron model on that data
        - plotting loss vs epochs (loss plot)
        - tqdm progress bar

5. feedforward
    - sigmoid neuron model for blob data
        - black code foramtter
        - blob data
        - classification of blob data using sigmoid neuron model
        - model class
        - visualizing prediction
    - simple feedforward model
        (3 neurons: one hidden layer with 2, one ouput layer with 1)
    - generic feedforward model
        (given : number of inputs, number of hidden layers, 
        number of neurons in each hidden layer)
        (one ouput neuron)
    - multiclass feedforward model
        (multiple neurons in output layer)
        - one hot encoding
        - softmax
        - cross entropy

6. visualization
    - update one weight only
        - gradient fn 
        - fit fn
    - hidden neruon function; weight - visualization
        - plotting "hidden neuron functions"
        - unpacking, packing
        - visualizing weights changing with epochs
        - heatmap, gif
    - scalar implementation of multiclass ff net

7. vector ff 
    - vectorization
        - time magic function
        - broadcasting, GPU
    - vector implementation of FF
        - copy, deepcopy
        - scalar implementation
        - vectorizing weights
        - vectorizing both weights and inputs
        - comparison

8. optimizaion 1
    - optimizaion algorithms
        - plotting change-in-parameters
    - visualization(animating parameter-changing-with-epochs over loss-"function" plot)
        - first plot surface/contour, then animate parameters with those axis-handlers
        - 3d surface + contour plot
        - contour plot
        - animation : changing plot-data using "line handler"
        - rc - display animation

9. optimizaion 2
    - more optimizaion algorithms
    - FF net - vectorized implementation with optimizaion algorithms
        - different optimizaion algorithms, hyperparameters - performance

10. inialization, activation
    - inialization, activation
        - plotting gradients (finding gradients using change/update in parameter)
        - multiple experiments in loops

11. regularization
    - changing model complexity
        - iris toy data
        - twinx (plotting)
    - l2 norm 
    - adding noise to training data
    - early stopping

-------------------------------------------------------------------------------------

12. torch
    - torch tensor
        - device, cuda
        - tensor operations
        - numpy <-> torch
        - tensor - ndarray bridge
    - comparing time
    - autograd
        - backward

13. ffn torch
    - ffn torch
        - unsqueeze
        - model, parameters, loss, fit, accuracy
    - nn.functional
    - nn.parameters
        - nn.Module
        - `__call__` magic method
        - independent fit function
    - nn.linear
    - optim
    - nn.sequential
    - fit function - template
    - run in gpu
        - fit function
        - model

14. cnn torch
    - dataset, dataloader
        - iter, next
        - demo loader
        - `__getitem__`
    - model inference
        - nn.Conv2d
        - detach
        - difference between torch tensor and numpy ndarray
        - torch.detach().numpy()
        - writing down the dimensions
    - training
    - gpu, visualization
        - moving to gpu, cpu (data, model)
        - accessing model blocks

15. cnn architectures
    - modifying model(vgg)
        - transforms (dataset)
        - torchvision.models , vgg 16
        - model class "attributes"
        - change whole sequential
        - delete batch
    - transfer weights
        - TORCH_HOME
        - pretrained weights
        - requires_grad = False (freeze)
    - checkpoint
        - copy.deepcopy
        - model.state_dict()
        - model.load_state_dict()
    - resnet(transfer)
        - f(x) + x
    - inception(transfer)
        - train mode, eval mode

16. cnn visualization
    - custom dataset
    - occlusion analysis
    - filter visualization

17. batchnorm, dropout
    - batchnorm
        - visualize distribution of all neuron values over datapoints
        - BatchNorm1d, BatchNorm2d
    - dropout
        - different forward pass in eval, train mode

18. rnn
    - RNNs
        - no opt.zero_grad (accumulating gradient)
        - IPython.display.clear_output (clear cell output)
        - lstm
        - gru

19. batching in sequence models
    - batching in sequence models
        - nn.RNN
        - function signature
        - batching
        - sequences to tensor with padding
        - packed sequence

20. encoder decoder, attention
    - encoder decoder
        - dataloader class (with getitem, len)
        - encoder decoder
        - encoder decoder with attention


         
