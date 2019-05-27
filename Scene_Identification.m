% Scene Identification Using Image Data
%% Description of the Data
% The dataset contains 4 scenes: Left Forward, Place, Right Forward
% The Laban symbols are photos that have been taken from different
% angles, positions. Howevever, they have a similarities in direction level 
% this a challenging task.

%% Load image data
% This assumes you have a directory: Demo1_BagOfFeatures
% with each scene in a subdirectory
imds = imageDatastore('C:\Users\michelle.de.bock\OneDrive\Documenten\Master Thesis\Afbeeldingen\Demo1_BagOfFeatures\Categories',...
    'IncludeSubfolders',true,'LabelSource','foldernames')              %#ok
imds.ReadFcn = @readAndResizeImages;
%% Display Class Names and Counts
tbl = countEachLabel(imds)                                             %#ok
categories = tbl.Label;

%% Display Sampling of Image Data
sample = splitEachLabel(imds,16);

montage(sample.Files(1:16));
title(char(tbl.Label(1)));

%% Show sampling of all data
for ii = 1:4
    sf = (ii-1)*16 +1;
    ax(ii) = subplot(2,2,ii);
    montage(sample.Files(sf:sf+3));
    title(char(tbl.Label(ii)));
end
% expandAxes(ax); % this is an optional feature, 

%% Pre-process Training Data: *Feature Extraction using Bag Of Words*
% Bag of features, also known as bag of visual words is one way to extract 
% features from images. To represent an image using this approach, an image 
% can be treated as a document and occurance of visual "words" in images
% are used to generate a histogram that represents an image.
%% Partition 25 images for training and 5 for testing
[training_set, test_set] = prepareInputFiles(imds);

%% Create Visual Vocabulary 
tic
bag = bagOfFeatures(training_set,...
    'VocabularySize',20,'PointSelection','Detector');
scenedata = double(encode(bag, training_set));
toc

%% Visualize Feature Vectors 
img = read(training_set(1), randi(training_set(1).Count));
featureVector = encode(bag, img);
subplot(4,2,1); imshow(img);
subplot(4,2,2); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(training_set(2), randi(training_set(2).Count));
featureVector = encode(bag, img);
subplot(4,2,3); imshow(img);
subplot(4,2,4); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(training_set(3), randi(training_set(3).Count));
featureVector = encode(bag, img);
subplot(4,2,5); imshow(img);
subplot(4,2,6); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

img = read(training_set(4), randi(training_set(4).Count));
featureVector = encode(bag, img);
subplot(4,2,7); imshow(img);
subplot(4,2,8); 
bar(featureVector);title('Visual Word Occurrences');xlabel('Visual Word Index');ylabel('Frequency');

%% Create a Table using the encoded features
SceneImageData = array2table(scenedata);
sceneType = categorical(repelem({training_set.Description}', [training_set.Count], 1));
SceneImageData.sceneType = sceneType;

%% Use the new features to train a model and assess its performance using 
%classificationLearner
trainingSceneData = double(encode(bag, training_set));
trainingSceneData = array2table(trainingSceneData,'VariableNames',trainedModel_EnsembleBaggedTrees.RequiredVariables);

% Test out accuracy on test set!

testSceneData = double(encode(bag, test_set));
testSceneData = array2table(testSceneData,'VariableNames',trainedModel_EnsembleBaggedTrees.RequiredVariables);
actualSceneType = categorical(repelem({test_set.Description}', [test_set.Count], 1));

predictedOutcome = trainedModel_EnsembleBaggedTrees.predictFcn(testSceneData);

correctPredictions = (predictedOutcome == actualSceneType);
validationAccuracy = sum(correctPredictions)/length(predictedOutcome) %#ok

% Visualize how the classifier works
figure(4);
ii = randi(size(test_set,2));
jj = randi(test_set(ii).Count);
img = read(test_set(ii),jj);

imshow(img)
%Add code here to invoke the trained classifier
imagefeatures = double(encode(bag, img));
%Find two closest matches for each feature
[bestGuess, score] = predict(trainedModel_EnsembleBaggedTrees.ClassificationEnsemble,imagefeatures);
%Display the string label for img
if strcmp(char(bestGuess),test_set(ii).Description)
	titleColor = [0 0.8 0];
else
	titleColor = 'r';
end
title(sprintf('Best Guess: %s; Actual: %s',...
	char(bestGuess),test_set(ii).Description),...
	'color',titleColor)

fprintf('The best guess is %s.\nThe correct answer is %s.\n',char(bestGuess),test_set(ii).Description);
fprintf('The validation accuracy is %d.\n',validationAccuracy);
 
%% Tabulate the results using a confusion matrix.
confMat = confusionmat(actualSceneType, predictedOutcome);
display(confMat); 
figure(5);
cm = confusionchart(actualSceneType, predictedOutcome, ...
    'Title','Confusion Chart', ...
    'RowSummary','row-normalized', ...
    'ColumnSummary','column-normalized');

