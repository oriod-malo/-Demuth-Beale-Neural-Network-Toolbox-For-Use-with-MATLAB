%% Chapter 6 - Control Systems

%% Types of controls in the Neural Network Toolbox
% • Model Predictive Control
% • NARMA-L2 (or Feedback Linearization) Control
% • Model Reference Control

%% Steps

%   1. The "System Identification" Step is the step where I develop a
%  neural network model of the plant I want to control. In each of the
%  three control types in the Neural Network Toolbox, the System
%  Identification step is the same.

%  2. The "Control Design" stage is different for each of the three control
%  types:
%       2.1. For the 'Model Predictive Control', the plant model is used to
%       predict the future behavior of the plant, and an optimization
%       algorithm is used to select the control input that optimizes future
%       performance.
%       2.2. For the 'NARMA-L2 Control', the controller is simply a
%       rearrangement of the plant model.
%       2.3. For the 'Model Reference Control', the controller is a Neural
%       Network that is trained to control a plant so that it follows a
%       reference model. The Neural Network plant model is used to assist
%       in the controller training.

%% Type - Model Predictive Control

% This controller uses a neural network model to predict fututre plant
% responses to potential control signals. An optimization algorithm then
% computes the control signals that optimize future plant performance, The
% neural network plant model is trained offline, in batch form, using any
% of the training algorithms discussed in Chapter 5. The controller,
% however, requries a significant amount of on-line computatuion, since an
% optimization algorithm is performed at each sample time to compute the
% optimal control input.

% Below follows the example of the Continous Stirret Tank Reactor (CSTR).

predcstr

%% Type - 