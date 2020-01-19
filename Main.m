clear all
close all
%% Data PreProcessing
rng(1337) %seed for random for consistend random number generation to make comparision of changes to neural network. can be commented out
fileID = fopen('TextFile.txt', 'a');
Data = csvread("GlassData.csv",1,1); 
for i = 1:9
    Data(:,i) = normalize(Data(:,i)); % normalize all data to be centered around zero with range from -1 to 1
end
initTrain = 1;
initTest = 1;   
initValid = 1;

% split up data randomly into rought 70% training 15% validation and 15%
% test
for i = 1:length(Data)
    r = rand();
    if (r < 0.7)
        if (initTrain == 1)
            initTrain = 0;
            trainData = Data(i,:);
        else
            trainData = [trainData;Data(i,:)];
        end
    elseif (r >= 0.7 && r <0.85)
        if (initTest == 1)
            initTest = 0;
            testData = Data(i,:);
        else
            testData = [testData;Data(i,:)];
        end
    elseif( r>=0.85)
        if (initValid == 1)
            initValid = 0;
            validData = Data(i,:);
        else
            validData = [validData;Data(i,:)];
        end
    end
end

%format validation data parse input and output. set output into 6 element
%array with an  indice set to one to indicate the glass type. the other
%elements are set to zero
validOut = zeros(length(validData),6);
for i = 1:length(validData)
    if validData(i,10) >4
       validData(i,10) = validData(i,10) -1;
    end
    validIn(i,:) = (validData(i,1:9));
    validOut(i,validData(i,10)) = 1;
end

for i = 1:9
    trainData(:,i) = (trainData(:,i));
end
%% Initialize Weights, ANN and inputvalues
% initialize number of nodes in each layer
h1Nodes = 10;
h2Nodes = 8;
outNodes = 6;

% initialize weights randomly (0-1)
h1weight = rand(10,h1Nodes); %weights leading to first hidden layer
h2weight = rand(h1Nodes,h2Nodes); %weights leading to second hidden layer
outweight = rand(h2Nodes,outNodes); %weights leading to output layer

% write weights
fprintf(fileID,'\nInitial weights for first hidden layer:\n');
fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f\n',h1weight(:,:));
fprintf(fileID,'\nInitial weights for second hidden layer:\n');
fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f\n',h2weight(:,:));
fprintf(fileID,'\nInitial weights for output layer:\n');
fprintf(fileID,'%f %f %f %f %f %f %f %f\n',outweight(:,:));

%set learning rate and momentum
ci = .05;%initial learning rate
c = ci; %updatable learning rate for later epochs
a = 1;
% initialize a secondary weight matrix for momentum calculations in
% gradient descent
oldoutweight = zeros(size(outweight));
oldh2weight = zeros(size(h2weight));
oldh1weight = zeros(size(h1weight));


n = 10000; % number of epochs

for j = 1:n
    % for each epoch randomly select 5 trials from each class. ensuring no
    % bias to due frequency. the set is used to train the epoch. also
    % formats epochexpectedvalues into array of 6 that with an indice set
    % to one to indicate glass type. the other elements are set to 0.
    epochInputValues = ones(30,10);
    epochExpectedValues = zeros(30,6);
    count = 1;
    for k = 1:7
        for l = 1:5
            A = trainData(trainData(:,10)==k,:);
            if(k ~= 4)
                randomselection = randi(size(A,1));
                epochInputValues(count,1:9) = A(randomselection,1:9);
                if (k < 4)
                epochExpectedValues(count,A(randomselection,10)) = 1;
                else
                epochExpectedValues(count,A(randomselection,10)-1) = 1;
                end
                count = count+1;
            end
        end
    end
    
    % apply a random noise to epoch input values to prevent over fitting. a
    % random gaussian distribution is used to create the error based on the
    % standard deviation of each feature
    for k = 1:length(epochInputValues)
        for l = 1:7
            epochInputValues(k,l) = epochInputValues(k,l) + (normrnd(0,.1*std(epochInputValues(:,l))));
        end
    end
    % back propagation algorithm with momentum.
    for i = 1:30
        %compute output values from each node
        h1Out = sigmoid((epochInputValues(i,:)*h1weight),1);
        h2Out = sigmoid((h1Out)*h2weight,1);
        output = sigmoid((h2Out)*outweight,1);
        %adjust weights leading to output nodes
        error = epochExpectedValues(i,:)-output;
        deltaOutweight = h2Out' *(c*error.*output.*(1-output));
        deltah2weight = h1Out' *((c.*error.*output.*(1-output))*outweight'.*h2Out.*(1-h2Out));
        deltah1weight = epochInputValues(i,:)'*((c.*error.*output.*(1-output))*outweight'.*h2Out.*(1-h2Out))*h2weight'.*h1Out.*(1-h1Out);
        outweight = outweight + deltaOutweight + oldoutweight;
        h2weight = h2weight + deltah2weight + oldh2weight;
        h1weight = h1weight + deltah1weight + oldh1weight;
        oldoutweight = deltaOutweight;
        oldh2weight = deltah2weight;
        oldh1weight = deltah1weight;
    end
    % calculate mean squared error using the validation data set
    for i = 1:length(validData)
    MSE(i) = sum(abs(validOut(i,:)-sigmoid(sigmoid(sigmoid([(validData(i,1:9)),1]*h1weight,1)*h2weight,1)*outweight,1)))^2;
    end
    epochMSE(j) = (1/(size(validData,1)))*sum(MSE);
    % if the mean squared error is below one break to prevent over fitting
    % and computation waste
    if epochMSE(j)<1
        break;
    end 
    % deccrease the learning rate value as epochs progress to reduce
    % overshooting
    c = ci*(n-j)/(n);
    % ouput various attributes to allow user to see progress of training
    [(j/n)*100 epochMSE(j) sum(epochMSE)/j c]
end

%% statistical analysis
% compute predictions using new weights
count = 0;
for i = 1:length(testData(:,10))
    if testData(i,10) > 4
    testData(i,10) = testData(i,10)-1;
    end
end
for i = 1:length(testData)
    [c,d] = max(sigmoid(sigmoid(sigmoid([(testData(i,1:9)),1]*h1weight,1)*h2weight,1)*outweight,1));
    D(i) = d;
end

% compute precision, recall and confusion matrix
confusion = confusionmat(testData(:,10),D);
recall = zeros(6,1);
precision = zeros(6,1);

for i = 1:length(confusion)
    recall(i) = confusion(i,i); 
    precision(i) = confusion(i,i);
end
for i = 1:length(confusion)
    recall (i) = recall(i)/sum(confusion(:,i));
    precision(i) = precision(i)/sum(confusion(i,:));
end

Meanrecall = sum(recall(~isnan(precision)))/6;
Meanprecision = sum(precision(~isnan(precision)))/6;
plot(D,'o');
hold on;
plot(testData(:,10),'*');
D(D==6) = D(D==6)+1;
D(D==5) = D(D==5) + 1;

% readjust the testdata back to skipping 4
testData(testData(:,10) == 6) =testData(testData(:,10)== 6) +1;
testData(testData(:,10) == 5) =testData(testData(:,10)== 5) +1;

%% write weights
fprintf(fileID,'\nFinal weights for first hidden layer:\n');
fprintf(fileID,'%f %f %f %f %f %f %f %f %f %f %f %f %f %f %f\n',h1weight(:,:));
fprintf(fileID,'\nFinal weights for second hidden layer:\n');
fprintf(fileID,'%f %f %f %f %f %f %f %f \n',h2weight(:,:));
fprintf(fileID,'\nFinal weights for output layer:\n');
fprintf(fileID,'%f %f %f %f %f %f\n',outweight(:,:));
fprintf(fileID,'\nConfusion Matrix:\n');
fprintf(fileID,'%i %i %i %i %i %i\n',confusion(:,:));
fprintf(fileID,'\nRecall:\n');
fprintf(fileID,'%f',Meanrecall);
fprintf(fileID,'\nPrecision:\n');
fprintf(fileID,'%f',Meanprecision);
fprintf(fileID,'\n TestData output');       
fprintf(fileID,'\nExpected       Predicted\n');
fprintf(fileID,'%i              %i\n',[testData(:,10),D']);
fclose(fileID);