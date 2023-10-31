%% Chapter 11 - Applications

%% Applin1: Linear Design

% Let's consider a signal T, which lasts 5 secondsm and is defined at a
% sampling rate of 40 samples per second. It would be defined as follows:  
time = 0:0.025:5; % from 0s to the 5ths (4 in total) with 1/40 step
T = sin(time*4*pi);
Q = length(T);

% At any given time step, the signal is given the last five values of the
% signal and it's expected to predict / generate the next value. The inputs
% P are found by delaying the signal T from one of the five time steps.

P = zeros(5,Q);
P(1,2:Q) = T(1,1:(Q-1));
P(2,3:Q) = T(1,1:(Q-2));
P(3,4:Q) = T(1,1:(Q-3));
P(4,5:Q) = T(1,1:(Q-4));
P(5,6:Q) = T(1,1:(Q-5));

% The plot of the sine wave
fig1 = figure(1);
hold on
plot(time,T,'-b');
title('Plot');
xlabel('time')
ylabel('T')
hold off

% Network Design

% Because the relationship between past and future values of the signal is
% not changing, the network can be designed as a Linear Network using the
% "newlind" function. 

% According to "help newlind", newlind(X,T) takes RxQ input matrix "X"
% and SxQ target matrix "T". In our case P is the input matrix and T is the
% target matrix. Since the problem above has five inputs, to solve the
% problem we need a neuron of five inputs.

net = newlind(P,T);
a = sim(net,P);

fig2 = figure(2);
hold on
plot(time,a,'-kx');
title('Plot after neuron')
xlabel('time');
ylabel('a');
hold off

fig3 = figure(3);
hold on
plot(time,T-a,'-');
title('Plot error');
xlabel('time');
ylabel('error e = T-a');
hold off


%% Applin2: Adaptive Prediction

% The signal T to be predicted lasts 6 seconds with a sampling rate of 20
% samples per second. However after 4 second the signal's frequency
% suddenly doubles.

time1 = 0:0.05:5; % 6 seconds, from 0 to 5, with 20 samples/s (1/20 step)
time2 = 4.05:0.024:6; % double frequency means half samples
time = [time1 time2];
T = [sin(time1*4*pi) sin(time2*8*pi)];
%%plot(time,T)
% Since we are training the network incrementally we better change to a
% sequence:
T = con2seq(T);

% 
P = T;

% NETWORK INITIALIZATION
lr = 0.1; % learning rate
delays = [1 2 3 4 5];
net = newlin(minmax(cat(2,P{:})),1,delays,lr);

[net,a,e] = adapt(net,P,T);

%% Appelm1: Amplitude Detection (APPELM = APPlication of ELMan networks)

% Elman Networks can be trained to recognize and produce both spatial and
% temporal patterns. An example of a problem where temporal patterns are
% recognized and classified with a spatial pattern is Amplitude Detection. 

% Amplitude Detection is not a difficult process and demonstrates Elman
% Network design process.

% Let's start by defining two sine waves with amplitudes 1 and 2 and then
% also two target outputs.

p1 = sin(1:20);
p2 = sin(1:20)*2;

% The target output for this waveform is their amplitude.

t1 = ones(1,20);
t2 = ones(1,20)*2;

% We combine them in a sequence where each wave repeats itself twice (why?)
% and then these waveforms are used to train the Elman Network.

p = [p1 p2 p1 p2];
t = [t1 t2 t1 t2];

% Since Inputs and targets need to be considered as sequence, we need to
% make the conversion from the Matrix Format.

Pseq = con2seq(p);
Tseq = con2seq(t);

% The problem requires that the Elman Network detect a single value (the
% signal) and output a single value (the amplitude) at each time step.
% Therefore the network must have one input element and one output neuron. 

% Input Element
R = 1;
% Layer 2 Output neuron
S2 = 1;

% The recurrent layer can have any number of neurons and more complex is a
% task the more neurons it requires. For this case 10 neurons will do.
S1 = 10;

% newelm => used to create initial weight matrices and bias vectors for a
% network with one input that can vary between -2 and +2 (bcs of sinewave
% with Amplitude = 2)

net = newelm([-2 2],[S1 S2],{'tansig','purelin'},'traingdx');

% ^ NOW WE CALL THE 'train' FUNCTION

[net, tr] = train(net,Pseq,Tseq);

a = sim(net,Pseq);

%%



%%


