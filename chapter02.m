%% Chapter 2 - Neuron Model and Network Architectures

%%
% 
% net.IW{1,1} = [1 2];
% net.b{1} = 0;
% 
% % p1 = [1;2]; p2 = [2;1]; p3 = [2;3]; p4 = [3,1];
% % these vectors are presented as a single matrix:
% 
% P = [1 2 2 3; 2 1 3 1];
% 
% %%
% A = sim(net,P)

%% Simulation With Sequential Inputs in a Dynamic Network

net = newlin([-1 1],1,[0 1]);
net.biasConnect = 0;

net.IW{1,1} = [1 2]; % to assign W = [1 2]

% Suppose that the input sequence is: p1 = 1 , p2 = 2 , p3 = 3 , p4 = 4
% Sequential inputs are presented to the network as elements of a cell array:

P = {1 2 3 4};

A = sim(net,P)

%% Simulation with Concurrent Inputs in a Dynamic Network

% If we were to apply the same inputs from the previous example 

net = newlin([-1 1],1,[0 1]);
net.biasConnect = 0;

net.IW{1,1} = [1 2]; % to assign W = [1 2]


% as a set of 
% concurrent inputs instead of a sequence of inputs, we would obtain a 
% completely different response. (Although, it is not clear why we would  
% want to do this with a dynamic network.) It would be as if each input   
% were applied concurrently to a separate parallel network. For the  
% previous example, if we use a concurrent set of inputs we have:

% p1 = 1 , p2 = 2 , p3 = 3 , p4 = 4

P = [1 2 3 4];
A = sim(net,P);

%%
net = newlin([-1 1],1,[0 1]);
net.biasConnect = 0;

net.IW{1,1} = [1 2]; % to assign W = [1 2]


% In certain special cases, we might want to simulate the network response to 
% several different sequences at the same time. In this case, we would want to 
% present the network with a concurrent set of sequences. For example, let’s say 
% we wanted to present the following two sequences to the network:

P = {[1 4] [2 3] [3 2] [4 1]};

A = sim(net,P)

%% TRAINING STYLES - Incremental Training (of Adaptive and Other Networks)

% Consider again the static network we used for our first example.
% We want to train it incrementally, 
% so that the weights and biases will be updated after each input is presented. 
% In this case we use the function adapt,
% and we present the inputs and targets as sequences.

% Suppose we want to train the network to create the linear function
% t = 2p1 + p2

% p1 = [1;2]; p2 = [2;1]; p3 = [2;3]; p4 = [3;1];
% the targets would be:
% t1 = 4 ; t2 = 5 ; t3 = 7 ; t4 = 7

% We first set up the network with zero initial weights and biases. We also set 
% the learning rate to zero initially, to show the effect of the incremental training.

net = newlin([-1 1;-1 1],1,0,0);
net.IW{1,1} = [0 0];
net.b{1} = 0;

% For incremental training we want to present the inputs and targets as 
% sequences:
P = {[1;2] [2;1] [2;3] [3;1]};
T = {4 5 7 7};

% We are now ready to train the network incrementally.

[net,a,e,pf] = adapt(net,P,T);
a
e
%%
% If we now set the learning rate to 0.1 we can see how the network is adjusted 
% as each input is presented:
net.inputWeights{1,1}.learnParam.lr=0.1;
net.biases{1,1}.learnParam.lr=0.1;
[net,a,e,pf] = adapt(net,P,T);
a
e

%% TRAINING STYLES - Incremental Training with Dynamic Networks

% We can also train dynamic networks incrementally. This would be the most
% common situation. We initialize the weights to zero and set the learning
% rate to 0.1

net = newlin([-1 1],1,[0 1],0.1);
net.IW{1,1} = [0 0];
net.biasConnect = 0;

% To train this network incrementally we present the inputs and targets as
% elements of cell array.
Pi = {1};
P = {2 3 4};
T = {3 5 7};

% Here we attempt to train the network to sum the current and previous
% inputs to create the current output. This is the same input sequence we
% used in the previous example using "sim" , except that we assign the
% first term in the sequence as the initial condition for the delay.

[net, a, e, pf] = adapt(net,P,T,Pi);
a
e

%% BATCH TRAINING - Batch Training with Static Networks

% Batch training can be done using either adapt or train, although train is 
% generally the best option, since it typically has access to more efficient training 
% algorithms. Incremental training can only be done with adapt; train can only 
% perform batch training.

net = newlin([-1 1;-1 1],1,0,0.1);
net.IW{1,1} = [0 0];
net.b{1} = 0;

P = [1 2 2 3; 2 1 3 1];
T = [4 5 7 7];

% When we call adapt, it will invoke trains (which is the default adaptation 
% function for the linear network) and learnwh (which is the default learning 
% function for the weights and biases).  Therefore, Widrow-Hoff learning is used.

[net,a,e,pf] = adapt(net,P,T);
a
e

%%
% Perform the same batch training using train.

net = newlin([-1 1;-1 1],1,0,0.1);
net.IW{1,1} = [0 0];
net.b{1} = 0;

P = [1 2 2 3; 2 1 3 1];
T = [4 5 7 7];
%%

% We will train it for only one epoch, since 
% we used only one pass of adapt. The default training function for the linear 
% network is trainc, and the default learning function for the weights and biases 
% is learnwh, so we should get the same results that we obtained using adapt in 
% the previous example, where the default adaptation function was trains.

net.inputWeights{1,1}.learnParam.lr = 0.1;
net.biases{1}.learnParam.lr = 0.1;
net.trainParam.epochs = 1;
net = trainc(net,P,T);

%% BATCH TRAINING - Batch Training With Dynamic Networks

net = newlin([-1 1],1,[0 1],0.02);
net.IW{1,1}=[0 0];
net.biasConnect=0;
net.trainParam.epochs = 1;
Pi = {1};
P = {2 3 4};
T = {3 5 6};

%%
net=train(net,P,T,Pi);
% » net.IW{1,1} you should get ans =     0.9000    0.6200
