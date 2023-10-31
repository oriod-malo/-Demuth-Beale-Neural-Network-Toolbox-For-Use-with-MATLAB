%% Chapter 3 - Perceptrons

%% Neuron Model

% The perceptron neuron produces a 1 if the net input into the transfer function 
% is equal to or greater than 0; otherwise it produces a 0.
% The hard-limit transfer function gives a perceptron the ability to classify input 
% vectors by dividing the input space into two regions. Specifically, outputs will 
% be 0 if the net input n is less than 0, or 1 if the net input n is 0 or greater.

%%                              Creating a Perceptron (net = newp(PR, S))

net = newp([0 2],1);
%%
inputweights = net.inputweights{1,1}
%%
biases = net.biases{1}

%
%
%
%

%%                              Simulation (sim)
%
net = newp([-2 2;-2 +2],1);
net.IW{1,1}= [-1 1];
net.b{1} = [1];

%%

% Test that parameters were set correctly
net.IW{1,1} %   should give ans = -1 1

net.b{1} %      should give ans = 1

%%
% Now let us see if the network responds to two signals, 
% one on each side of the perceptron boundary
%%
p1 = [1;1];
a1 = sim(net,p1)

% should be 1

%%
p2 = [1;-1]
a2 = sim(net,p2)

% should be 0

%%

%  could present the two inputs in a sequence and get the outputs in a sequence as well

p3 = {[1;1],[1;-1]}
a3 = sim(net,p3)

%%                              Initialization (init)
%
%%
net = newp([-2 2;-2 +2],1);
wts = net.IW{1,1} % wts = 0 0       at this point
bias = net.b{1} % bias = 0          at this point


net.IW{1,1} = [3,4];
net.b{1} = 5;
%%
wts = net.IW{1,1}
bias = net.b{1}
%%
% We use init to reset the weights so wts and bias are again [0,0] and 0

net = init(net)
wts = net.IW{1,1}
bias = net.b{1}
%%
% We can change the way a perceptron is initialized with the "init"
% function. For example, we can redefine the inputweights.initFcn and
% biases.initFcn as 'rands' in order to have random weights and biases at
% each initialization

net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
net = init(net);

wts = net.IW{1,1}
bias = net.b{1}


%%

%% Learning Rules

% A "learning rule" is a procedure used for modifying the weights and
% biases of a network. This procedure is also called a "traning algorithm".
% The learning rule is applied to train the network to perform a particular
% task.

% Learning rules in the Matlab DNN Toolbox are either Supervised Learning
% Rules or Unsupervised Learning Rules.

% In SUPERVISED LEARNING, the learning rule is provided with a set of
% examples (the training set) of proper network behavior:
%
% {p1,t1},{p2,t2},...,{pq,tq}
%
% where pq is an input to the network, and tq is the corresponding correct 
% (target) output.

% In UNSUPERVISED LEARNING, the weights and biases are modified in  
% response to network inputs only. There are no target outputs available.


%% Perceptron Learning Rule (learnp)

% Start with a single neuron having a input vector with just two elements.

net = newp([-2 2;-2 +2],1);

net.b{1} = [0];
w = [1 -0.8];
net.IW{1,1} = w;

%The input target pair is given by
p = [1; 2];
t = [1];
%%
a = sim(net,p)
a
e = t-a
%%
% dw = learnp(w,p,[],[],[],[],e,[],[],[])

% w = w + dw

%% Training (train)

% read the text from the book

net = newp([-2 2;-2 +2],1);
p =[2; 2];
t =[0];

net.trainParam.epochs = 1;
net = train(net,p,t);
%%
net = newp([-2 2;-2 +2],1);
net.trainParam.epochs = 1;
p = [[2;2] [1;-2] [-2;2] [-1;1]];
t =[0 1 0 1];
net = train(net,p,t);

%%
a = sim(net,p)

% gives a = 0 0 1 1 - not what we wanted

%%
% we train again for 4 epochs
%%
net = newp([-2 2;-2 +2],1);
net.trainParam.epochs = 4;
p = [[2;2] [1;-2] [-2;2] [-1;1]];
t =[0 1 0 1];
net = train(net,p,t);
%%
a = sim(net,p)
% a = 0 1 0 1
% e = a1-t1 a2-t2 a3-t3 a4-t4 = 0 0 0 0 we got what we wanted
%% Graphical User Interface

%%
nntool