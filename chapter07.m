%% Chapter 7 - Radial Basis Functions

%% Neuron Model & Network Architecture
% (read in the book - it's theory)

%% Exact Design (newrbe)

% The exact desing (meaning 0 error) on training vectors can be performed
% with the function "newrbe":

%   net = newrbe(P,T,SPREAD)    where

% P and T are the input and target vectors respectively, while SPREAD is a
% spread constant.

% "newrbe" creates as many radbas neurons as there are input vectors in P,
% and sets the first-layer weights to P'. If there are Q input vectors
% there are Q neurons

%% More Efficient Design (newrb)

% The function "newrb" creates the radial basis network one neuron at a
% time. Neurons are added to the network until either:
%   a) the sum square error falls below the GOAL value
%   b) the maximum number of neurons has been reached

%% Generalized Regression Networks (GRNN)

P = [4 5 6];
T = [1.5 3.6 6.7];

net = newgrnn(P,T);

P = 4.5;
v = sim(net,P)

%% Probabilistic Neural Networks (PNN)
% Probabilistic neural networks can be used for classification problems. 
% When an  input is presented, the first layer computes distances from the 
% input vector to the training input vectors, and produces a vector whose 
% elements indicate how close the input is to a training input. 
% The second layer sums these contributions for each class of inputs to 
% produce as its net output a vector of probabilities.

% Finally, a compete transfer function on the output of the second layer
% picks the maximum of these probabilities, and produces a 1 for that class 
% and a 0 for the other classes. The architecture for this system is shown 
% below.

P = [0 0;1 1;0 3;1 4;3 1;4 1;4 3]'
Tc = [1 1 2 2 3 3 3];

T = ind2vec(Tc);

net = newpnn(P,T);
Y = sim(net,P)
%%
Yc = vec2ind(Y)

P2 = [1 4;0 1;5 2]'