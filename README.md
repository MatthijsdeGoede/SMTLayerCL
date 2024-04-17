# SMTLayerCL

Implementation of SMTLayer with Curriculum Learning in Pytorch with Z3.

This code is the python companion to the **Enhancing the Training of Neuro-Symbolic Neural
Networks through Curriculum Learning based on
Symbolic Domain Knowledge** paper by Matthijs de Goede and is an adaptation of the original SMTLayer implementation: https://github.com/cmu-transparency/smt-layer

To install the requirements, run: 
```pip install -r requirements.txt```

The original SMTLayer was just adapted to incorporate symbol correctness, and can be found in
```smtlayer/smtlayer.py```

The class implementing the calculation of the heuristic that corresponds to ```Algorithm 1``` in the paper can be found in
```scorer.py```

The original script to run the Visual MNIST task in ```scripts/mnist_addition.py``` has been adapted to incorporate symbol correctness and to use curriculum learning.
The curriculum construction in accordance with ```Algorithm 2``` from the paper is implemented using custom data loader,
for which most important details can be found in the ```get_curriculum_dataloader``` method.

The results of the experiments, along with the plots can be found under ```results```

The folder ```scripts``` further contains a few helper methods that were used to process the results.
```scripts/plotter.py``` was used to generate the plots from the paper. ```scripts/tabler.py``` was used to construct ```Table 1``` 
and ```scripts/time_pre_processing.py``` was used to calculate the average calculation times for the heuristic score during execution. 

To re-run our experiments run ```scripts/runner.py```. Here one can also schedule alternative experiments, using a custom configuration based on the named parameters from the ```run```
method in ```scripts/mnist_addition.py```
