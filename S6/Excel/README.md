#  leveraging Neural Network by Excel 

In this guide, we're going to walk through the steps of creating a neural network that utilizes backpropagation, all within Excel. By leveraging Excel's built-in formulas and functions, we'll construct a straightforward and functional neural network model. Throughout the tutorial, we aim to demystify the fundamental principles that drive neural networks and the backpropagation algorithm, making sure each explanation is accessible and easy to grasp.


**Activation Function:**
An activation function in a neural network serves as a critical non-linear transformation that allows the network to learn complex patterns. It decides whether a neuron should be activated or not, determining the output of that neuron given a set of inputs. Common activation functions include the sigmoid, which squashes output values to a range between 0 and 1; the hyperbolic tangent (tanh), which outputs values between -1 and 1; and the Rectified Linear Unit (ReLU), which outputs the input directly if it is positive, otherwise, it will output zero. Activation functions are not only pivotal for learning complex patterns but also for introducing non-linearity to the model, making it capable of learning from and performing tasks on a wide range of data.

**Forward Propagation:**
Forward propagation is the initial phase of the learning process in a neural network where input data is passed through the network's layers to generate an output. In this phase, the input data is processed layer by layer, from the input layer through the hidden layers and finally to the output layer. At each neuron, an activation function is applied to the weighted sum of the inputs, and the result is forwarded to the next layer. The final output is then used to calculate the error of the network's prediction, which forms the basis for the subsequent backward propagation phase.

**Backward Propagation:**
Backward propagation, or backpropagation, is the second phase of the learning process in neural networks, following forward propagation. It is where the network learns by adjusting the weights of the neurons. Backpropagation computes the gradient of the loss function (a measure of the error) with respect to each weight by the chain rule, propagating the error backward through the network. Starting from the output layer and moving backward to the hidden layers, the algorithm calculates the contributions of each neuron to the error and updates the weights to minimize the loss function. This process allows the network to improve its predictions during training.

![alt text](image-1.png)


h1 = w1*i1 + w2*i2		
h2 = w3*i1 + w4*i2		
a_h1 = σ(h1) = 1/(1 + exp(-h1))		
a_h2 = σ(h2)		
o1 = w5*a_h1 + w6*a_h2		
o2 = w7*a_h1 + w8*a_h2		
a_o1 = σ(o1)		
a_o2 = σ(o2)		
E_total = E1 + E2		
E1 = ½ * (t1 - a_o1)²		
E2 = ½ * (t2 - a_o2)²		





# Learning Rate:
<p float="left">
  <figure>
    <img src="https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/85b4dba8-f2e5-462a-92c0-3998260bb22c" width="450" />
    <figcaption>LR: 0.1</figcaption>
  </figure>
  <figure>
    <img src="https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/ffacf37d-12a0-4b97-8cce-b260f45e25f0" width="450" />
    <figcaption>LR: 0.2</figcaption>
  </figure>
  <figure>
    <img src="https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/50df3711-ec00-4f17-9967-08cf2ecc6604" width="450" />
    <figcaption>LR: 0.5</figcaption>
  </figure>
  <figure>
    <img src="https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/b67c7d4f-d187-4bc8-b7f8-1bea53b26861" width="450" />
    <figcaption>LR: 0.8</figcaption>
  </figure>
  <figure>
    <img src="https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/c04711c7-f24c-45c0-aa21-b946655a815e" width="450" />
    <figcaption>LR: 1.0</figcaption>
  </figure>
  <figure>
    <img src="https://github.com/Omkar1634/ERA_V2_Omkar/assets/64948764/4f4fbffc-6ac6-49b9-9a1d-ddb48ec113f7" width="450" />
    <figcaption>LR: 2.0</figcaption>
  </figure>
</p>
