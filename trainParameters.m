function trainPar = trainParameters()
    
    trainPar.x = [1 0 0 ; 1 0 1 ; 1 1 0 ; 1 1 1];                    % Training input data where the first value of 1 represents the bias term
    trainPar.y = [0 ; 1 ; 1 ; 0];                                    % Labelled / Desired output data
    trainPar.noi = size(trainPar.x , 1);                             % Number of input
    trainPar.w = rand(5 , 3);                                        % Unknown / Weight parameters which randomly initialized
    trainPar.y_hat = zeros(size(trainPar.x , 1) , 1);                % Initialize estimated output 
    trainPar.mu = 0.2;                                               % Initialize the learning rate
    trainPar.it = 100000;                                            % Initialize the iteration number
    trainPar.e = zeros(size(trainPar.x , 1) , 1);                    % Initialize the instant error
    trainPar.es = zeros(size(trainPar.x , 1) , trainPar.it);         % Initialize the estimated error
    trainPar.ys_hat = zeros(size(trainPar.x , 1) , trainPar.it);     % Initialize the estimated output
    
end
