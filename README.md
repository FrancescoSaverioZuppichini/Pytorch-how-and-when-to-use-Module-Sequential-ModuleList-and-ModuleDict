
# Pytorch: how and when to use Module, Sequential, ModuleList and ModuleDict
### Effective way to share, reuse and break down the complexity of your models

Updated at Pytorch 1.5

You can find the code [here](https://github.com/FrancescoSaverioZuppichini/Pytorch-how-and-when-to-use-Module-Sequential-ModuleList-and-ModuleDict)

[Pytorch](https://pytorch.org/) is an open source deep learning frameworks that provide a smart way to create ML models. Even if the documentation is well made, I still see that most people don't write well and organized code in PyTorch.

Today, we are going to see how to use the three main building blocks of PyTorch: `Module, Sequential and ModuleList`. We are going to start with an example and iteratively we will make it better.

All these four classes are contained into `torch.nn`


```python
import torch.nn as nn

# nn.Module
# nn.Sequential
# nn.Module
```

## Module: the main building block

The [Module](https://pytorch.org/docs/stable/nn.html?highlight=module) is the main building block, it defines the base class for all neural network and you MUST subclass it. 

Let's create a classic CNN classifier as example:


```python
import torch.nn.functional as F

class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(64 * 28 * 28, 1024)
        self.fc2 = nn.Linear(1024, n_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = x.view(x.size(0), -1) # flat
        
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        
        return x
```


```python
model = MyCNNClassifier(1, 10)
print(model)
```

    MyCNNClassifier(
      (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (fc1): Linear(in_features=50176, out_features=1024, bias=True)
      (fc2): Linear(in_features=1024, out_features=10, bias=True)
    )


This is a very simple classifier with an encoding part that uses two layers with 3x3 convs + batchnorm + relu and a decoding part with two linear layers. If you are not new to PyTorch you may have seen this type of coding before, but there are two problems.

If we want to add a layer we have to again write lots of code in the `__init__` and in the `forward` function. Also, if we have some common block that we want to use in another model, e.g. the 3x3 conv + batchnorm + relu, we have to write it again.

## Sequential: stack and merge layers

[Sequential](https://pytorch.org/docs/stable/nn.html?highlight=sequential#torch.nn.Sequential) is a container of Modules that can be stacked together and run at the same time.

You can notice that we have to store into `self` everything. We can use `Sequential` to improve our code. 


```python
class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_c, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
```


```python
model = MyCNNClassifier(1, 10)
print(model)
```

    MyCNNClassifier(
      (conv_block1): Sequential(
        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_block2): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (decoder): Sequential(
        (0): Linear(in_features=50176, out_features=1024, bias=True)
        (1): Sigmoid()
        (2): Linear(in_features=1024, out_features=10, bias=True)
      )
    )


Much Better uhu?

Did you notice that `conv_block1` and `conv_block2` looks almost the same? We could create a function that reteurns a `nn.Sequential` to even simplify the code!


```python
def conv_block(in_f, out_f, *args, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        nn.ReLU()
    )
```

Then we can just call this function in our Module


```python
class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.conv_block1 = conv_block(in_c, 32, kernel_size=3, padding=1)
        
        self.conv_block2 = conv_block(32, 64, kernel_size=3, padding=1)

        
        self.decoder = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
```


```python
model = MyCNNClassifier(1, 10)
print(model)
```

    MyCNNClassifier(
      (conv_block1): Sequential(
        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (conv_block2): Sequential(
        (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): ReLU()
      )
      (decoder): Sequential(
        (0): Linear(in_features=50176, out_features=1024, bias=True)
        (1): Sigmoid()
        (2): Linear(in_features=1024, out_features=10, bias=True)
      )
    )


Even cleaner! Still `conv_block1` and `conv_block2` are almost the same! We can merge them using `nn.Sequential`


```python
class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_c, 32, kernel_size=3, padding=1),
            conv_block(32, 64, kernel_size=3, padding=1)
        )

        
        self.decoder = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
```


```python
model = MyCNNClassifier(1, 10)
print(model)
```

    MyCNNClassifier(
      (encoder): Sequential(
        (0): Sequential(
          (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (decoder): Sequential(
        (0): Linear(in_features=50176, out_features=1024, bias=True)
        (1): Sigmoid()
        (2): Linear(in_features=1024, out_features=10, bias=True)
      )
    )


`self.encoder` now holds booth `conv_block`. We have decoupled logic for our model and make it easier to read and reuse. Our `conv_block` function can be imported and used in another model.

## Dynamic Sequential: create multiple layers at once

What if we can to add a new layers in `self.encoder`, hardcoded them is not convinient:

```python
self.encoder = nn.Sequential(
            conv_block(in_c, 32, kernel_size=3, padding=1),
            conv_block(32, 64, kernel_size=3, padding=1),
            conv_block(64, 128, kernel_size=3, padding=1),
            conv_block(128, 256, kernel_size=3, padding=1),

        )
```

Would it be nice if we can define the sizes as an array and automatically create all the layers without writing each one of them? Fortunately we can create an array and pass it to `Sequential`


```python
class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, n_classes):
        super().__init__()
        self.enc_sizes = [in_c, 32, 64]
        
        conv_blocks = [conv_block(in_f, out_f, kernel_size=3, padding=1) 
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        
        self.encoder = nn.Sequential(*conv_blocks)

        
        self.decoder = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
```


```python
model = MyCNNClassifier(1, 10)
print(model)
```

    MyCNNClassifier(
      (encoder): Sequential(
        (0): Sequential(
          (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (decoder): Sequential(
        (0): Linear(in_features=50176, out_features=1024, bias=True)
        (1): Sigmoid()
        (2): Linear(in_features=1024, out_features=10, bias=True)
      )
    )


Let's break it down. We created an array `self.enc_sizes` that holds the sizes of our encoder. Then we create an array `conv_blocks` by iterating the sizes. Since we have to give booth a in size and an outsize for each layer we `zip`ed the size'array with itself by shifting it by one. 

Just to be clear, take a look at the following example:


```python
sizes = [1, 32, 64]

for in_f,out_f in zip(sizes, sizes[1:]):
    print(in_f,out_f)
```

    1 32
    32 64


Then, since `Sequential` does not accept a list, we decompose it by using the `*` operator.

Tada! Now if we just want to add a size, we can easily add a new number to the list. It is a common practice to make the size a parameter.


```python
class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, enc_sizes, n_classes):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        
        conv_blocks = [conv_block(in_f, out_f, kernel_size=3, padding=1) 
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        
        self.encoder = nn.Sequential(*conv_blocks)

        
        self.decoder = nn.Sequential(
            nn.Linear(64 * 28 * 28, 1024),
            nn.Sigmoid(),
            nn.Linear(1024, n_classes)
        )

        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
```


```python
model = MyCNNClassifier(1, [32,64, 128], 10)
print(model)
```

    MyCNNClassifier(
      (encoder): Sequential(
        (0): Sequential(
          (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (2): Sequential(
          (0): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (decoder): Sequential(
        (0): Linear(in_features=50176, out_features=1024, bias=True)
        (1): Sigmoid()
        (2): Linear(in_features=1024, out_features=10, bias=True)
      )
    )


We can do the same for the decoder part


```python
def dec_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Sigmoid()
    )

class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, enc_sizes, dec_sizes,  n_classes):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        self.dec_sizes = [64 * 28 * 28, *dec_sizes]

        conv_blocks = [conv_block(in_f, out_f, kernel_size=3, padding=1) 
                       for in_f, out_f in zip(self.enc_sizes, self.enc_sizes[1:])]
        
        self.encoder = nn.Sequential(*conv_blocks)

        
        dec_blocks = [dec_block(in_f, out_f) 
                       for in_f, out_f in zip(self.dec_sizes, self.dec_sizes[1:])]
        
        self.decoder = nn.Sequential(*dec_blocks)
        
        self.last = nn.Linear(self.dec_sizes[-1], n_classes)

        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.view(x.size(0), -1) # flat
        
        x = self.decoder(x)
        
        return x
```


```python
model = MyCNNClassifier(1, [32,64], [1024, 512], 10)
print(model)
```

    MyCNNClassifier(
      (encoder): Sequential(
        (0): Sequential(
          (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
        (1): Sequential(
          (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
          (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU()
        )
      )
      (decoder): Sequential(
        (0): Sequential(
          (0): Linear(in_features=50176, out_features=1024, bias=True)
          (1): Sigmoid()
        )
        (1): Sequential(
          (0): Linear(in_features=1024, out_features=512, bias=True)
          (1): Sigmoid()
        )
      )
      (last): Linear(in_features=512, out_features=10, bias=True)
    )


We followed the same pattern, we create a new block for the decoding part, linear + sigmoid, and we pass an array with the sizes. We had to add a `self.last` since we do not want to activate the output

Now, we can even break down our model in two! Encoder + Decoder


```python
class MyEncoder(nn.Module):
    def __init__(self, enc_sizes):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[conv_block(in_f, out_f, kernel_size=3, padding=1) 
                       for in_f, out_f in zip(enc_sizes, enc_sizes[1:])])

        def forward(self, x):
            return self.conv_blocks(x)
        
class MyDecoder(nn.Module):
    def __init__(self, dec_sizes, n_classes):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f) 
                       for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Linear(dec_sizes[-1], n_classes)

    def forward(self, x):
        return self.dec_blocks()
    
    
class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, enc_sizes, dec_sizes,  n_classes):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        self.dec_sizes = [self.enc_sizes[-1] * 28 * 28, *dec_sizes]

        self.encoder = MyEncoder(self.enc_sizes)
        
        self.decoder = MyDecoder(self.dec_sizes, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.flatten(1) # flat
        
        x = self.decoder(x)
        
        return x
```


```python
model = MyCNNClassifier(1, [32,64], [1024, 512], 10)
print(model)
```

    MyCNNClassifier(
      (encoder): MyEncoder(
        (conv_blocks): Sequential(
          (0): Sequential(
            (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
          (1): Sequential(
            (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): ReLU()
          )
        )
      )
      (decoder): MyDecoder(
        (dec_blocks): Sequential(
          (0): Sequential(
            (0): Linear(in_features=50176, out_features=1024, bias=True)
            (1): Sigmoid()
          )
          (1): Sequential(
            (0): Linear(in_features=1024, out_features=512, bias=True)
            (1): Sigmoid()
          )
        )
        (last): Linear(in_features=512, out_features=10, bias=True)
      )
    )


Be aware that `MyEncoder` and `MyDecoder` could also be functions that returns a `nn.Sequential`. I prefer to use the first pattern for models and the second for building blocks.

By diving our module into submodules it is easier to **share** the code, **debug** it and **test** it.

## ModuleList : when we need to iterate

`ModuleList` allows you to store `Module` as a list. It can be useful when you need to iterate through layer and store/use some information, like in U-net.

The main difference between `Sequential` is that `ModuleList` have not a `forward` method so the inner layers are not connected. Assuming we need each output of each layer in the decoder, we can store it by:


```python
class MyModule(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(in_f, out_f) for in_f, out_f in zip(sizes, sizes[1:])])
        self.trace = []
        
    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            self.trace.append(x)
        return x
```


```python
model = MyModule([1, 16, 32])
import torch

model(torch.rand((4,1)))

[print(trace.shape) for trace in model.trace]
```

    torch.Size([4, 16])
    torch.Size([4, 32])





    [None, None]



## ModuleDict: when we need to choose

What if we want to switch to `LearkyRelu` in our `conv_block`? We can use `ModuleDict` to create a dictionary of `Module` and dynamically switch `Module` when we want


```python
def conv_block(in_f, out_f, activation='relu', *args, **kwargs):
    
    activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()]
    ])
    
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        activations[activation]
    )
```


```python
print(conv_block(1, 32,'lrelu', kernel_size=3, padding=1))
print(conv_block(1, 32,'relu', kernel_size=3, padding=1))
```

    Sequential(
      (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): LeakyReLU(negative_slope=0.01)
    )
    Sequential(
      (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (2): ReLU()
    )


## Final implementation

Let's wrap it up everything!


```python
def conv_block(in_f, out_f, activation='relu', *args, **kwargs):
    activations = nn.ModuleDict([
                ['lrelu', nn.LeakyReLU()],
                ['relu', nn.ReLU()]
    ])
    
    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        activations[activation]
    )

def dec_block(in_f, out_f):
    return nn.Sequential(
        nn.Linear(in_f, out_f),
        nn.Sigmoid()
    )

class MyEncoder(nn.Module):
    def __init__(self, enc_sizes, *args, **kwargs):
        super().__init__()
        self.conv_blocks = nn.Sequential(*[conv_block(in_f, out_f, kernel_size=3, padding=1, *args, **kwargs) 
                       for in_f, out_f in zip(enc_sizes, enc_sizes[1:])])
        
        def forward(self, x):
            return self.conv_blocks(x)
        
class MyDecoder(nn.Module):
    def __init__(self, dec_sizes, n_classes):
        super().__init__()
        self.dec_blocks = nn.Sequential(*[dec_block(in_f, out_f) 
                       for in_f, out_f in zip(dec_sizes, dec_sizes[1:])])
        self.last = nn.Linear(dec_sizes[-1], n_classes)

    def forward(self, x):
        return self.dec_blocks()
    
    
class MyCNNClassifier(nn.Module):
    def __init__(self, in_c, enc_sizes, dec_sizes,  n_classes, activation='relu'):
        super().__init__()
        self.enc_sizes = [in_c, *enc_sizes]
        self.dec_sizes = [32 * 28 * 28, *dec_sizes]

        self.encoder = MyEncoder(self.enc_sizes, activation=activation)
        
        self.decoder = MyDecoder(dec_sizes, n_classes)
        
    def forward(self, x):
        x = self.encoder(x)
        
        x = x.flatten(1) # flat
        
        x = self.decoder(x)
        
        return x

```


```python
model = MyCNNClassifier(1, [32,64], [1024, 512], 10, activation='lrelu')
print(model)
```

    MyCNNClassifier(
      (encoder): MyEncoder(
        (conv_blocks): Sequential(
          (0): Sequential(
            (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.01)
          )
          (1): Sequential(
            (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): LeakyReLU(negative_slope=0.01)
          )
        )
      )
      (decoder): MyDecoder(
        (dec_blocks): Sequential(
          (0): Sequential(
            (0): Linear(in_features=1024, out_features=512, bias=True)
            (1): Sigmoid()
          )
        )
        (last): Linear(in_features=512, out_features=10, bias=True)
      )
    )


## Conclusion
So, in summary.

- Use `Module` when you have a big block compose of multiple smaller blocks
- Use `Sequential` when you want to create a small block from layers
- Use `ModuleList` when you need to iterate through some layers or building blocks and do something
- Use `ModuleDict` when you need to parametise some blocks of your model, for example an activation function

That's all folks!

Thank you for reading
