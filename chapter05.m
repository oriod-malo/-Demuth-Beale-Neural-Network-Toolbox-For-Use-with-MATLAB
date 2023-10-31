%% Chapter 5 - Feedforward Network, Simulation and Training

%%
help newff
 
%      newff(P,T,S) takes,
%        P  - RxQ1 matrix of Q1 representative R-element input vectors.
%        T  - SNxQ2 matrix of Q2 representative SN-element target vectors.
%        Si  - Sizes of N-1 hidden layers, S1 to S(N-1), default = [].
%              (Output layer size SN is determined from T.)
%      and returns an N layer feed-forward backprop network.
%  
%      newff(P,T,S,TF,BTF,BLF,PF,IPF,OPF,DDF) takes optional inputs,
%        TFi - Transfer function of ith layer. Default is 'tansig' for
%              hidden layers, and 'purelin' for output layer.
%        BTF - Backprop network training function, default = 'trainlm'.
%        BLF - Backprop weight/bias learning function, default = 'learngdm'
%        PF  - Performance function, default = 'mse'.
%        IPF - Row cell array of input processing functions.
%              Default is {'fixunknowns','remconstantrows','mapminmax'}.
%        OPF - Row cell array of output processing functions.
%              Default is {'remconstantrows','mapminmax'}.
%        DDF - Data division function, default = 'dividerand';
%      and returns an N layer feed-forward backprop network.
 

%%
% The following command creates a two-layer network:
net = newff([-1 2; 0 5],[3,1],{'tansig','purelin'},'traingd');
% + There is one input vector with two elements:
%   - The values for the first element of the input vector range between -1
%   and 2 [-1 2 ;...
%   - The values of the second element of the input vector range between 0
%   and 5 ... ; 0 5]
% + There are three neurons in the first layer and one neuron in the second
% (output) layer [3,1]
%   - The transfer function in the first layer is tan-sigmoid
%   - The output layer transfer function is linear

net = init(net); % Initializing Weights (init)

p = [1 3 2;2 4 1]; 

% The function sim simulates a network. sim takes the network input p, and
% the network object net, and returns the network outputs a.
a = sim(net,p) 

%% Batch Gradient Descent (traingd)

p = [ -1 -1 2 2 ; 0 5 0 5];
t = [ -1 -1 1 1];

net = newff(minmax(p), [3,1], {'tansig','purelin'}, 'traingd');

net.trainParam.show = 50;
net.trainParam.lr = 0.05;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;

[net,tr] = train(net,p,t);

a = sim(net,p)

%% Batch Gradient Descent with Momentum (traingdm)

p = [-1 -1 2 2; 0 5 0 5];
t = [-1 -1 1 1];

net = newff(minmax(p), [3,1], {'tansig','purelin'}, 'traingdm');
net.trainParam.show = 50;
net.trainParam.lr = 0.05;
net.trainParam.mc = 0.9;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;
[net,tr] = train(net,p,t);

a = sim(net,p)

%% Variable Learning Rate (traingda, traingdx)

p = [-1 -1 2 2;0 5 0 5];
t = [-1 -1 1 1];

net = newff(minmax(p), [3,1], {'tansig','purelin'}, 'traingda');
net.trainParam.show = 50;
net.trainParam.lr = 0.05;
net.trainParam.lr_inc = 1.05;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;
[net,tr]=train(net,p,t);
a = sim(net,p)

%% Resilient Backpropagation (trainrp)

p = [-1 -1 2 2 ; 0 5 0 5];
t = [-1 -1 1 1];

net = newff(minmax(p), [3,1], {'tansig','purelin'}, 'trainrp');
net.trainParam.show = 10;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;
[net,tr]=train(net,p,t);
a = sim(net,p)

%% 

%% Preprocessing and Postprocessing

p = [-1 -1 2 2 ; 0 5 0 5];
t = [-1 -1 1 1];

net = newff(minmax(p), [3,1], {'tansig','purelin'}, 'trainrp');
net.trainParam.show = 10;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;

% The DNN Matlab toolbox gives the possibility to preprocess min and max p
% and t so they can always fall under the [-1,1] range. This can be done
% through the function "premnmx". For example:

% RUN THE "Resilient Backpropagation" SECTION FIRST
[pn, minp, maxp, tn, mint, maxt] = premnmx(p,t);
net = train(net, pn, tn);
an = sim(net,pn);

% If I preprocessed my targets with premnmx then I need to post-process it
% with the function postmnmx.
a = postmnmx(an,mint,maxt);

% "an" corresponds to the normalzied targets tn, "a" to the un-normalized
% ones.

%% Mean and Stand. Dev. (prestd, poststd, trastd)

p = [-1 -1 2 2; 0 5 0 5];
t = [-1 -1 1 1];

net = newff(minmax(p), [3,1], {'tansig','purelin'}, 'traingdm');
net.trainParam.show = 50;
net.trainParam.lr = 0.05;
net.trainParam.mc = 0.9;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;

% Another approach for scaling inputs is normalizing the standard deviation
% and the mean of the training set. This is done with prestd.

[pn, meanp, stdp, tn, meant, stdt] = prestd(p,t);
net = train(net, pn, tn);
an = sim(net,pn);
a = poststd(an,meant,stdt);

%% Principal Component Analysis (prepca, trapca)

% In some situations when the dimension of the input vector is large and
% the components of the vector are highly correlated (hence redundant) it
% is useful to reduce the dimension of the input vectors

% This can be done through component analysis. Component Analysis has three
% effects:
%   1 - orthogonalizes the components of each vector
%   2 - orders the orthogonal components so that those with the largest
%   variation come first.
%   3 - it eliminates those components that contribute the least to the
%   variation in the data set

% [pn,meanp,stdp] = prestd(p);
% [ptrans,transMat] = prepca(pn,0.02);
% 
%% Post-Training Analysis (postreg)

p = [-1 -1 2 2; 0 5 0 5];
t = [-1 -1 1 1];

net = newff(minmax(p), [3,1], {'tansig','purelin'}, 'traingdm');
net.trainParam.show = 50;
net.trainParam.lr = 0.05;
net.trainParam.mc = 0.9;
net.trainParam.epochs = 300;
net.trainParam.goal = 1e-5;

% The performance of a trained network can be measured to some extent by the 
% errors on the training, validation and test sets, but it is often useful to 
% investigate the network response in more detail. One option is to perform a 
% regression analysis between the network response and the corresponding 
% targets. The routine postreg is designed to perform this analysis.

a = sim(net,p);
[m,b,r] = postreg(a,t)

%% Sample Training Session

