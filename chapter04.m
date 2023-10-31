%% Chapter 4 - Linear Filters
%
% Linear filters treated in this chapter are similar to the Perceptron with
% the difference that their transfer function is linear not hard-limiting.
% This enables them to take any value not just 0 or 1.

% Linear networks and perceptrons can solve only linearly separable
% problems

%% Creating a Linear Neuron (newlin)

net = newlin([-1 1; -1 1],1);

% The first matrix of arguments specify the range of the two scalar inputs
% The last argument, 1, says that the network has a single output.

%%
W = net.IW{1,1}
b= net.b{1}

%%
% I can give to the Weights and the Bias any value I want
net.IW{1,1} = [2 3];
W = net.IW{1,1}

net.b{1} =[-4];
b = net.b{1}
%%
% I can simulate the network as follows:

p=[5;6]; %input vector

a = sim(net,p) % should get a = 24 in the command window

%% Linear System Design (newlind)

% Linear systems can also be designed directly if input and target pairs
% are known.

% Specific network values for weights and biases can be obtained to
% minimize the mean square error by using the funcion "newlind"

% Suppose the inputs are:
P = [1 2 3];
T = [2.0 4.1 5.9];

net = newlind(P,T);

Y = sim(net,P)
% Should get Y =     2.0500    4.0000    5.9500

%% Linear filter

% Linear filters are created by combining tappped delay lines with linear
% networks.

% If we want the linear layer that, given input sequence P and initial
% input delays Pi, outputs the target sequence T, we can write (as an
% example):

P = {1 2 1 3 3 2};
Pi = {1 3};
T = {5 6 4 20 7 8};

% then
net = newlind(P,T,Pi);
Y = sim(net,P,Pi)

% Network outputs are not exactly equal to the targets but they are
% close, which means that the mean square error has been minimized

%% Linear Classification (train)

P = [2 1 -2 -1;2 -2 2 1];
t = [0 1 0 1];
net = newlin( [-2 2; -2 2],1);
net.trainParam.goal= 0.1;
[net, tr] = train(net,P,t);

weights = net.iw{1,1}
bias = net.b(1)
%%
A = sim(net, P) % Should get A =  0.0282 0.9672 0.2741 0.4320
err = t - sim(net,P) % Should get err =  -0.0282 0.0328 -0.2741 0.5680

