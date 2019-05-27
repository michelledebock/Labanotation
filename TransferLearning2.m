%% Create an imageDataStore to read images
location = 'C:\Users\michelle.de.bock\OneDrive\Documenten\Master Thesis\Afbeeldingen\Demo2_TransferLearning\Categories';
imds = imageDatastore(location,'IncludeSubfolders',1,...
    'LabelSource','foldernames');

[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

numTrainImages = numel(imdsTrain.Labels);
idx = randperm(numTrainImages,16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTrain,idx(i));
    imshow(I)
end

%% AlexNet
net = alexnet;
analyzeNetwork(net)

%% input layer
inputSize = net.Layers(1).InputSize

%% Replace layers
layersTransfer = net.Layers(1:end-3);

numClasses = numel(categories(imdsTrain.Labels));

layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

%% Train network
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');


netTransfer = trainNetwork(augimdsTrain,layers,options);


%% Classify Validation Images
[YPred,scores] = classify(netTransfer,augimdsValidation);


idx = randperm(numel(imdsValidation.Files),1);
figure
for i = 1:1
    subplot(2,2,i)
    I = readimage(imdsValidation,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end


YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

display(accuracy);


%% Tabulate the results using a confusion matrix.
confMat = confusionmat(YValidation, YPred);
display(confMat); 
figure(5);
cm = confusionchart(YValidation, YPred, ...
    'Title','Confusion Chart', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');


