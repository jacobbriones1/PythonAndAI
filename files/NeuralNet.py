def sigmoid(x):
  return 1/(1+np.exp(-x))

class Neuron:
  def __init__(self, weights, bias=0,name=None):
    self.name=name
    self.weights = np.array(weights)
    self.bias = bias
    self.output = None
    self.num_inputs = len(list(weights))
    self.parameters = {"name":self.name,
                       "inputs":[],
                       "weights":[self.weights],
                            "bias":[self.bias],
                            "output":[],
                            'num_inputs':self.num_inputs,
                            }
    self.output = None

  def forward(self, input):
    self.output = sigmoid(self.weights.dot(input)+self.bias)
    self.parameters['output'].append(self.output)
    return sigmoid(self.weights.dot(input)+self.bias)

  def update_weights(self, weights):
    self.weights = np.array(weights)
    self.parameters['weights'].append([self.weights])

  def info(self):
    j=0
    for key in self.parameters.keys():
      print('\t'+key +': '+ str(self.parameters[key])+',')


class Layer:
  def __init__(self, num_inputs, num_outputs,name=""):
    self.neurons = []
    self.name=name
    for i in range(num_outputs):
      self.neurons.append(Neuron(list(np.random.uniform(0,1,size=(num_inputs))) ,name=name+'_neuron'+str(i+1)))
    self.size=[num_inputs,num_outputs]
    self.inputs = []
    self.outputs = []
    self.parameters = {'inputs':[],
                       'neurons':[n for n in self.neurons],
                       'weights':np.array([n.weights for n in self.neurons]),
                       'biases':np.array([n.bias for n in self.neurons]),
                       'outputs':[n.output for n in self.neurons if n.output!=None] }

  def forward(self, inputs):
    self.inputs.append(np.array(inputs))
    self.outputs = [neuron.forward(inputs) for neuron in self.neurons]
    self.parameters['outputs'].append(self.outputs)
    self.parameters['inputs'].append(self.inputs)
    return self.outputs

  def info(self):
    print('layer: '+self.name+": {")
    i = 0
    for neuron in self.neurons:
      neuron.info()
      if i < len(self.neurons)-1:
        print()
      else:
        print("}\n\n")
      i=i+1

class SequentialModel:
  def __init__(self, layers=[],name=""):
    self.layers=layers
    self.name=name
    self.inputs=[]
    self.outputs=[]

  def add(self, layer:Layer):
    self.layers.append(layer)

  def forward(self, input):
    self.inputs.append(input)
    for layer in layers:
      out = layer.forward(input)
      input = out
    self.outputs.append(out)

  def summary(self):
    print("----------------------------")

    print("| Model : {}\t   |".format(self.name))
    print("----------------------------")
    print("| Layer\t\tShape      |")
    print("----------------------------")

    for l in self.layers:
      print("{}\t({},{})\n".format(l.name,l.size[0],l.size[1]))

