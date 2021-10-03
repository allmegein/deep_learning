% This m.file trains a multi layer NN (MLP) for the XOR problem
clear , close all;
clc;

trainPar = trainParameters();       % Upload the training parameters
w = trainPar.w;                     % Upload the unknown parameters
mu = trainPar.mu;                   % Upload the learning coefficient
y_hat = trainPar.y_hat;             % Upload the output
es = trainPar.es;                   % Upload the allocated estimated error
e = trainPar.e;                     % Upload the instant error
ys_hat = trainPar.ys_hat;           % Upload the allocated estimated output

for i = 1 : trainPar.it
    for j = 1 : trainPar.noi
        
        % Calculate the output of the first hidden layer neuron
        H1 = w(1,:) * trainPar.x(j,:)';

        % Apply a threshold for estimated output of first hidden layer neuron
        X1(2) = sigmoid(H1);

        % Calculate the output of the second hidden layer neuron
        H2 = w(2,:)* trainPar.x(j,:)';

        % Apply a threshold for estimated output of second hidden layer neuron
        X1(3) = sigmoid(H2);

        % Assign a bias term for output layer 
        X1(1) = 1;
        
        H2_1 = w(3,:) * X1';
        X2(2) = sigmoid(H2_1);
        
        H2_2 = w(4,:) * X1';
        X2(3) = sigmoid(H2_2);
        
        X2(1) = 1;
        

        % Calculate the output of the output layer neuron
        X3 = w(5,:) * X2';

        % Apply a threshold for output layer neuron
        y_hat(j) = sigmoid(X3);

        % Calculate the error of the output and store it
        e(j,:) = trainPar.y(j) - y_hat(j);

        % Calculate the error for each neuron using delta rule
        delta3_1 = y_hat(j) * (1 - y_hat(j)) * (trainPar.y(j) - y_hat(j));  
        delta2_1 = X2(2) * (1 - X2(2)) * w(5,2) * delta3_1;                 
        delta2_2 = X2(3) * (1 - X2(3)) * w(5,3) * delta3_1;                 
        
        delta1_1 = X1(2) * (1 - X1(2)) * w(3,2) * delta2_1;
        delta1_2 = X1(2) * (1 - X1(2)) * w(4,2) * delta2_2;
        delta1_3 = X1(3) * (1 - X1(3)) * w(3,3) * delta2_1;
        delta1_4 = X1(3) * (1 - X1(3)) * w(4,3) * delta2_2;
        
        % Update the unknown parameters
        w(1,:) = w(1,:) + mu * trainPar.x(j,:) * (delta1_1 + delta1_2);
        w(2,:) = w(2,:) + mu * trainPar.x(j,:) * (delta1_3 + delta1_4);
        w(3,:) = w(3,:) + mu * X1 * delta2_1;
        w(4,:) = w(4,:) + mu * X1 * delta2_2;
        w(5,:) = w(5,:) + mu * X2 * delta3_1;
        
    end

    % Store the estimated output 
    ys_hat(:,i) = y_hat;

    % Store the error history 
    es(:,i) = e;
    
end

% Plot the estimated output for the XOR gate
figure(1),plot(1:length(ys_hat),ys_hat(1,:),'r','LineWidth',2),hold on,
plot(1:length(ys_hat),ys_hat(2,:),'b','LineWidth',2),
plot(1:length(ys_hat),ys_hat(3,:),'g','LineWidth',2),
plot(1:length(ys_hat),ys_hat(4,:),'y','LineWidth',2), hold off;
xlabel('iteration'),ylabel('output');
title('Estimated Outputs for the XOR Gate');

% Plot the error history for the XOR gate
figure(),plot(1:length(es),es(1,:),'r','LineWidth',2),hold on,
plot(1:length(es),es(2,:),'b','LineWidth',2),
plot(1:length(es),es(3,:),'g','LineWidth',2),
plot(1:length(es),es(4,:),'y','LineWidth',2), hold off;
xlabel('iteration'),ylabel('error');
title('Training Error for the XOR Gate');